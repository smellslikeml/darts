import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import numpy as np
import torch

from darts import concatenate
from darts.models import XPatchModel
from darts.tests.conftest import tfm_kwargs
from darts.utils import timeseries_generation as tg
from darts.utils.likelihood_models.torch import GaussianLikelihood


class TestXpatchModels:
    np.random.seed(42)
    torch.manual_seed(42)

    def test_creation(self):
        # valid model
        model = XPatchModel(
            input_chunk_length=12,
            output_chunk_length=6,
            patch_len=4,
            stride=2,
        )
        assert model.input_chunk_length == 12
        assert model.ma_type == "ema"

        # `patch_len` larger than `input_chunk_length`
        with pytest.raises(ValueError):
            XPatchModel(
                input_chunk_length=12,
                output_chunk_length=6,
                patch_len=16,
                stride=8,
            )

        # `output_chunk_length` too small for the trend stream pooling
        with pytest.raises(ValueError):
            XPatchModel(
                input_chunk_length=12,
                output_chunk_length=1,
                patch_len=4,
                stride=2,
            )

        # invalid moving average type / padding
        with pytest.raises(ValueError):
            XPatchModel(
                input_chunk_length=12,
                output_chunk_length=6,
                patch_len=4,
                stride=2,
                ma_type="sma",
            )
        with pytest.raises(ValueError):
            XPatchModel(
                input_chunk_length=12,
                output_chunk_length=6,
                patch_len=4,
                stride=2,
                padding_patch="start",
            )

        # smoothing factors must be strictly between 0 and 1
        for alpha, beta in [(0.0, 0.3), (1.0, 0.3), (0.3, 0.0), (0.3, 1.0)]:
            with pytest.raises(ValueError):
                XPatchModel(
                    input_chunk_length=12,
                    output_chunk_length=6,
                    patch_len=4,
                    stride=2,
                    alpha=alpha,
                    beta=beta,
                )

    def test_fit(self):
        large_ts = tg.constant_timeseries(length=100, value=1000)
        small_ts = tg.constant_timeseries(length=100, value=10)

        for kwargs in [
            {"ma_type": "ema"},
            {"ma_type": "dema"},
            {"ma_type": "reg"},
            {"ma_type": "ema", "use_reversible_instance_norm": True},
        ]:
            # Test basic fit and predict
            model = XPatchModel(
                input_chunk_length=12,
                output_chunk_length=6,
                patch_len=4,
                stride=2,
                n_epochs=20,
                random_state=42,
                **tfm_kwargs,
                **kwargs,
            )
            model.fit(large_ts[:94])
            pred = model.predict(n=2).values()[0]

            # Test whether model trained on one series is better than one trained on another
            model2 = XPatchModel(
                input_chunk_length=12,
                output_chunk_length=6,
                patch_len=4,
                stride=2,
                n_epochs=20,
                random_state=42,
                **tfm_kwargs,
                **kwargs,
            )
            model2.fit(small_ts[:94])
            pred2 = model2.predict(n=2).values()[0]
            assert abs(pred2 - 10) < abs(pred - 10)

            # test short predict
            pred3 = model2.predict(n=1)
            assert len(pred3) == 1

    def test_fit_predict_shapes(self):
        # univariate
        ts = tg.sine_timeseries(length=60, value_frequency=0.1)
        model = XPatchModel(
            input_chunk_length=12,
            output_chunk_length=6,
            patch_len=4,
            stride=2,
            n_epochs=2,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(ts)
        pred = model.predict(n=6)
        assert pred.width == 1
        assert len(pred) == 6
        assert np.all(np.isfinite(pred.values()))

        # multivariate: each component is forecast with its own (channel-independent) stream
        ts_multi = concatenate(
            [
                tg.sine_timeseries(length=60, value_frequency=0.1),
                tg.sine_timeseries(length=60, value_frequency=0.05, value_amplitude=2),
            ],
            axis="component",
        )
        model = XPatchModel(
            input_chunk_length=12,
            output_chunk_length=6,
            patch_len=4,
            stride=2,
            n_epochs=2,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(ts_multi)
        pred = model.predict(n=6)
        assert pred.width == 2
        assert len(pred) == 6
        assert np.all(np.isfinite(pred.values()))

        # autoregressive prediction with n > output_chunk_length
        pred = model.predict(n=8)
        assert len(pred) == 8

    def test_past_covariates(self):
        ts = tg.sine_timeseries(length=60, value_frequency=0.1)
        past_cov = tg.sine_timeseries(length=60, value_frequency=0.05)

        model = XPatchModel(
            input_chunk_length=12,
            output_chunk_length=6,
            patch_len=4,
            stride=2,
            n_epochs=2,
            random_state=42,
            **tfm_kwargs,
        )
        assert model.supports_past_covariates
        assert not model.supports_future_covariates
        assert not model.supports_static_covariates

        model.fit(ts, past_covariates=past_cov)
        pred = model.predict(n=6, past_covariates=past_cov)
        assert pred.width == 1
        assert len(pred) == 6
        assert np.all(np.isfinite(pred.values()))

    def test_likelihood_fit(self):
        ts = tg.constant_timeseries(length=24)

        model = XPatchModel(
            input_chunk_length=12,
            output_chunk_length=6,
            patch_len=4,
            stride=2,
            n_epochs=1,
            random_state=42,
            likelihood=GaussianLikelihood(),
            **tfm_kwargs,
        )
        model.fit(ts)
        # sampled from distribution
        pred = model.predict(n=6, num_samples=20)
        assert pred.n_samples == 20
        assert pred.width == 1

        # direct distribution parameter prediction
        pred = model.predict(n=6, num_samples=1, predict_likelihood_parameters=True)
        assert pred.width == 2
        assert pred.n_samples == 1

    def test_backprop(self):
        """Short backward pass / loss decrease check on CPU (device-agnostic port)."""
        ts = tg.sine_timeseries(length=60, value_frequency=0.1)
        model = XPatchModel(
            input_chunk_length=12,
            output_chunk_length=6,
            patch_len=4,
            stride=2,
            n_epochs=1,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(ts)
        module = model.model

        # reproducible synthetic batch: (batch, input_chunk_length, components)
        t = torch.arange(12, dtype=torch.float32)
        x = torch.stack([torch.sin(t * phi) for phi in (0.3, 0.5, 0.7, 0.9)]).unsqueeze(
            -1
        )
        y = torch.roll(x, -6, dims=1)  # forecast = last 6 steps shifted

        module.train()
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            out = module((x, None, None, None))[..., 0]  # (batch, out_len, comps)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # gradients flow through both streams
        assert module.network.fc8.weight.grad is not None
        assert torch.all(torch.isfinite(module.network.fc8.weight.grad))
        # the loss decreases over the manual training steps
        assert np.mean(losses[-3:]) < losses[0]
