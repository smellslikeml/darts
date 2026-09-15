"""
xPatch
------
"""

# The exponential seasonal-trend decomposition layers (EMA / DEMA) and the dual-stream network
# (``nn.Modules``) were adapted from the author's reference implementation of xPatch:
# https://github.com/stitsyuk/xPatch
#
# The reference implementation is licensed under the Apache License, Version 2.0:
# https://github.com/stitsyuk/xPatch/blob/main/LICENSE
#
# Adaptations compared to the reference implementation:
# - device-agnostic tensors: the reference implementation hardcodes ``.to("cuda")`` in the moving
#   average layers, which prevents training/inference on CPU; here all tensors are created on the
#   device of the input batch
# - reversible instance normalization is delegated to Darts' ``RINorm``
#   (``use_reversible_instance_norm``) instead of the reference's internal ``RevIN`` layer
# - the forecast head outputs ``nr_params`` parameters per target component to support Darts'
#   probabilistic ``likelihood`` models
# - past covariates are supported as additional channel-independent input features

from typing import Literal

import torch
import torch.nn as nn

from darts.logging import raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample

MA_TYPES = [
    "ema",
    "dema",
    "reg",
]


class _ExponentialMovingAverage(nn.Module):
    """Exponential moving average (EMA) block to highlight the trend of time series"""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_length, channels)
        # vectorized implementation with a cumulative sum: each output is a weighted average of
        # the current and all previous inputs, normalized by the sum of the weights
        dtype = x.dtype
        _, input_length, _ = x.shape
        powers = torch.flip(
            torch.arange(input_length, dtype=torch.double, device=x.device), dims=(0,)
        )
        weights = torch.pow(1.0 - self.alpha, powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, input_length, 1)
        divisor = divisor.reshape(1, input_length, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(dtype=dtype)


class _DoubleExponentialMovingAverage(nn.Module):
    """Double exponential moving average (DEMA) block to highlight the trend of time series"""

    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_length, channels)
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            s = self.alpha * x[:, t, :] + (1.0 - self.alpha) * (s_prev + b)
            b = self.beta * (s - s_prev) + (1.0 - self.beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)


class _ExponentialDecomposition(nn.Module):
    """Exponential seasonal-trend series decomposition block"""

    def __init__(self, ma_type: str, alpha: float, beta: float):
        super().__init__()
        if ma_type == "ema":
            self.ma = _ExponentialMovingAverage(alpha)
        elif ma_type == "dema":
            self.ma = _DoubleExponentialMovingAverage(alpha, beta)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_average = self.ma(x)
        seasonal = x - moving_average
        return seasonal, moving_average


class _XPatchNetwork(nn.Module):
    """Dual-stream (non-linear patch CNN + linear MLP) forecasting network"""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        nr_params: int,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.nr_params = nr_params

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(
            self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatenation
        self.fc8 = nn.Linear(pred_len * 2, pred_len * nr_params)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # s, t: (batch, input_length, channels); `s` is the seasonal (non-linear) stream input and
        # `t` the trend (linear) stream input
        s = s.permute(0, 2, 1)  # to (batch, channels, input_length)
        t = t.permute(0, 2, 1)  # to (batch, channels, input_length)

        # Channel split for channel independence
        batch_size = s.shape[0]
        channels = s.shape[1]
        input_length = s.shape[2]
        s = torch.reshape(s, (batch_size * channels, input_length))
        t = torch.reshape(t, (batch_size * channels, input_length))

        # Non-linear Stream
        # Patching
        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: (batch * channels, patch_num, patch_len)

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatenation
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatenation and likelihood parameters
        # (batch * channels, pred_len * nr_params) -> (batch, pred_len, channels, nr_params)
        x = torch.reshape(x, (batch_size, channels, self.pred_len, self.nr_params))
        x = x.permute(0, 2, 1, 3)

        return x


class _XPatchModule(PLForecastingModule):
    """
    xPatch module
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        ma_type: str,
        alpha: float,
        beta: float,
        **kwargs,
    ):
        """PyTorch module implementing the xPatch architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional past covariate)
        output_dim
            Number of output components in the target
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        patch_len
            The length of each patch extracted from the input sequence.
        stride
            The stride used to extract the patches from the input sequence.
        padding_patch
            Whether to pad the input sequence at the end (by repeating the last value) to get an
            additional patch (``"end"``), or not to pad (``"none"``).
        ma_type
            The type of moving average used for the seasonal-trend decomposition: ``"ema"``,
            ``"dema"``, or ``"reg"`` (no decomposition).
        alpha
            The smoothing factor of the (double) exponential moving average.
        beta
            The trend smoothing factor of the double exponential moving average (only used with
            ``ma_type="dema"``).
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_dim)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, output_dim, nr_params)`
            Tensor containing the output of the xPatch module.
        """

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params

        # Exponential Seasonal-Trend Decomposition
        self.ma_type = ma_type
        if ma_type in ("ema", "dema"):
            self.decomposition = _ExponentialDecomposition(ma_type, alpha, beta)

        self.network = _XPatchNetwork(
            seq_len=self.input_chunk_length,
            pred_len=self.output_chunk_length,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            nr_params=nr_params,
        )

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        """
        x_in
            comes as tuple `(x, x_future, x_static, future_target)` where `x` is the past target,
            past covariates and historic future covariate chunk. Input dimensions are
            `(n_samples, n_time_steps, n_variables)`
        """
        x, _, _, _ = x_in  # x: (batch, in_len, in_dim)

        if self.ma_type == "reg":
            # no decomposition: pass the input directly to both streams
            x = self.network(x, x)
        else:
            seasonal, trend = self.decomposition(x)
            x = self.network(seasonal, trend)

        # discard the outputs of the (channel-independent) past covariate channels: the target
        # components are the first `output_dim` input features
        return x[:, :, : self.output_dim, :]


class XPatchModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Literal["end", "none"] = "end",
        ma_type: Literal["ema", "dema", "reg"] = "ema",
        alpha: float = 0.3,
        beta: float = 0.3,
        **kwargs,
    ):
        """An implementation of the xPatch model, as presented in [1]_.

        xPatch is a dual-stream architecture for multivariate time series forecasting: the input is
        decomposed into a seasonal and a trend component with an exponential (EMA) or double
        exponential (DEMA) moving average. The seasonal component is processed by a non-linear
        stream (patching followed by depthwise/pointwise convolutions) while the trend component is
        processed by a linear MLP stream. Both streams are channel-independent and their outputs are
        concatenated to form the forecast.

        This implementation adds the optional use of past covariates (known for
        `input_chunk_length` points before prediction time, processed as additional
        channel-independent input features), as well as support for probabilistic forecasting.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it). Must be at least as
            large as `patch_len` (and at least 2 with `ma_type="dema"`).
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support). Must be at least 2.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        patch_len
            The length of each patch extracted from the input sequence. Default: 16.
        stride
            The stride used to extract the patches from the input sequence. Default: 8.
        padding_patch
            Whether to pad the input sequence at the end (by repeating the last value) to get an
            additional patch (``"end"``), or not to pad (``"none"``). Default: ``"end"``.
        ma_type
            The type of moving average used for the seasonal-trend decomposition: ``"ema"``
            (exponential moving average), ``"dema"`` (double exponential moving average), or
            ``"reg"`` (no decomposition: the raw input is passed to both streams). Default: ``"ema"``.
        alpha
            The smoothing factor of the (double) exponential moving average, in ``(0, 1)``.
            Default: 0.3.
        beta
            The trend smoothing factor of the double exponential moving average, in ``(0, 1)``.
            Only used with `ma_type="dema"`. Default: 0.3.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.
            It is only applied to the features of the target series and not the covariates. If ``True``,
            applies ``RINorm`` with default hyperparameters. If a dictionary, defines the hyperparameters to construct
            the ``RINorm``. Supported parameters are ``{"affine": bool, "eps": float}``. Default: ``False``.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the process ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        random_state
            Controls the randomness of the weights initialization and reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUs.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`__

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.
        enable_finetuning
            Enables model fine-tuning. Only effective if not ``None``.
            If a bool, specifies whether to perform full fine-tuning / training (all parameters are updated) or keep
            all parameters frozen. If a dict, specifies which parameters to fine-tune. Must only contain one key-value
            record. Can be used to:

            - Unfreeze specific parameters, while keeping everything else frozen:
              ``{"unfreeze": ["param.name.patterns.*"]}``
            - Freeze specific parameters, while keeping everything else unfrozen:
              ``{"freeze": ["param.name.patterns.*"]}``

            Default: ``None``.

        References
        ----------
        .. [1] Stitsyuk, V. (2025).
               "xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition".
               AAAI Conference on Artificial Intelligence (2025), arXiv preprint arXiv:2412.17323,
               https://arxiv.org/abs/2412.17323. Reference implementation (Apache-2.0):
               https://github.com/stitsyuk/xPatch
        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import XPatchModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # predict 6 pressure values using the 24 past values of pressure
        >>> model = XPatchModel(
        >>>     input_chunk_length=24,
        >>>     output_chunk_length=6,
        >>>     n_epochs=20,
        >>> )
        >>> model.fit(target)
        >>> pred = model.predict(6)

        .. note::
            This simple usage example produces poor forecasts. In order to obtain better performance, user should
            transform the input data, increase the number of epochs, use a validation set, optimize the hyper-
            parameters, ...
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.ma_type = ma_type
        self.alpha = alpha
        self.beta = beta

        if ma_type not in MA_TYPES:
            raise_log(
                ValueError(
                    f"Invalid `ma_type`: {ma_type}. Must be one of `{MA_TYPES}`."
                ),
            )
        if padding_patch not in ("end", "none"):
            raise_log(
                ValueError(
                    f"Invalid `padding_patch`: {padding_patch}. Must be one of `['end', 'none']`."
                ),
            )
        if patch_len < 1 or stride < 1:
            raise_log(
                ValueError(
                    f"`patch_len` and `stride` must be positive integers, received "
                    f"`patch_len={patch_len}` and `stride={stride}`."
                ),
            )
        if input_chunk_length < patch_len:
            raise_log(
                ValueError(
                    f"`input_chunk_length` ({input_chunk_length}) must be at least as large as "
                    f"`patch_len` ({patch_len})."
                ),
            )
        if ma_type == "dema" and input_chunk_length < 2:
            raise_log(
                ValueError(
                    "`input_chunk_length` must be at least 2 with `ma_type='dema'`."
                ),
            )
        if output_chunk_length < 2:
            raise_log(
                ValueError(
                    f"`output_chunk_length` ({output_chunk_length}) must be at least 2."
                ),
            )
        if not 0.0 < alpha < 1.0:
            raise_log(
                ValueError(f"`alpha` ({alpha}) must be strictly between 0.0 and 1.0."),
            )
        if not 0.0 < beta < 1.0:
            raise_log(
                ValueError(f"`beta` ({beta}) must be strictly between 0.0 and 1.0."),
            )

    def _create_model(self, train_sample: TorchTrainingSample) -> torch.nn.Module:
        # samples are made of (past target, past cov, historic future cov, future cov, static cov, future_target)
        (past_target, past_covariates, _, _, _, _) = train_sample
        input_dim = past_target.shape[1] + (
            past_covariates.shape[1] if past_covariates is not None else 0
        )
        output_dim = past_target.shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _XPatchModule(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            patch_len=self.patch_len,
            stride=self.stride,
            padding_patch=self.padding_patch,
            ma_type=self.ma_type,
            alpha=self.alpha,
            beta=self.beta,
            **self.pl_module_params,
        )
