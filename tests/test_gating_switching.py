"""Zero-API-call MockBackend verification of SAGE gating and switching.

Tests the routing mechanism without any LLM API calls.
"""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sage.config import Config
from sage.data.datasets import synthetic
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent
from sage.heavy.nexus_pipeline import MockBackend, NexusPipeline, ForecastInputs
from sage.pipeline import SAGEPipeline
from sage.verify.decision import decide
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder


def cfg(**kwargs):
    return replace(Config(window=4, hidden=3, embedding_dim=8), **kwargs)


class CountingBackend(MockBackend):
    """MockBackend that records every complete() call."""
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.call_roles = []

    def complete(self, role, payload, prompts=None, prompt_version="v1", task_instruction=""):
        self.call_count += 1
        self.call_roles.append(role)
        return super().complete(role, payload, prompts=prompts,
                                prompt_version=prompt_version,
                                task_instruction=task_instruction)


def make_pipeline(config, backend=None, cache_dir=None, gate_bias=None):
    """Build a SAGEPipeline with optional gate head bias override."""
    if backend is None:
        backend = CountingBackend()
    heavy = NexusPipeline(backend, cache_dir, prompt_version="v1", max_rul=config.max_rul)
    small = SmallModel(config)
    gate = FuzzyDecisionAgent(config)
    reflection = SelfReflectionAgent(config)
    encoder = TraceEncoder(config)

    if gate_bias is not None:
        with torch.no_grad():
            gate.head.bias.fill_(gate_bias)

    return SAGEPipeline(config, small, gate, heavy, reflection, encoder), backend


def make_inputs(config, values=None):
    if values is None:
        values = [[float(j) for j in range(config.features)] for _ in range(config.window)]
    return ForecastInputs(
        values=values,
        timestamps=[f"t-{i}" for i in range(config.window)],
        context=[""] * config.window,
        feature_names=[f"sensor_{i}" for i in range(config.features)],
        task="Predict RUL at the final cycle.",
        horizon=config.horizon,
        target_index=None,
    )


class GatingSwitchingTests(unittest.TestCase):
    """Verify routing correctness with MockBackend (zero API calls)."""

    def setUp(self):
        self.config = cfg(tau1=0.60, tau2=0.05)
        self.x = torch.from_numpy(
            synthetic(self.config).x[:1].astype(np.float32)
        )

    def test_1_small_model_only_routing(self):
        """High gate confidence -> bypass heavy path, source='small', zero calls."""
        backend = CountingBackend()
        pipeline, _ = make_pipeline(self.config, backend=backend, gate_bias=+10.0)
        inputs = make_inputs(self.config)

        with torch.no_grad():
            result = pipeline(self.x, inputs, mode="sage")

        self.assertEqual(result.source, "small")
        self.assertIsNone(result.heavy)
        self.assertEqual(backend.call_count, 0, "Heavy path must not be invoked when gate accepts small")
        self.assertEqual(len(backend.call_roles), 0)
        # Final prediction equals small-model prediction
        small_pred = pipeline.small(self.x).prediction[0].tolist()
        np.testing.assert_allclose(result.prediction, small_pred)

    def test_2_heavy_path_invocation(self):
        """Low gate confidence -> escalate, exactly 4 backend calls."""
        backend = CountingBackend()
        pipeline, _ = make_pipeline(self.config, backend=backend, gate_bias=-10.0)
        inputs = make_inputs(self.config)

        with torch.no_grad():
            result = pipeline(self.x, inputs, mode="sage")

        self.assertIn(result.source, ("heavy", "fusion"))
        self.assertIsNotNone(result.heavy)
        self.assertEqual(backend.call_count, 4, "Exactly four agent calls expected")
        self.assertEqual(sorted(backend.call_roles), ["context", "macro", "micro", "synthesis"])

    def test_3_d2_heavy_versus_fusion(self):
        """D2 decision with controlled confidence values."""
        c = cfg(tau1=0.60, tau2=0.05)
        y_s = [10.0]
        y_l = [12.0]

        # Case A: q_s=0.1, q_l=0.8 -> delta = -0.7 <= tau2(0.05) -> accept heavy
        dec_a = decide(y_s, y_l, 0.1, 0.8, c)
        self.assertTrue(dec_a.accept_heavy)
        self.assertEqual(dec_a.source, "heavy")
        self.assertEqual(dec_a.prediction, [12.0])
        self.assertAlmostEqual(dec_a.delta, -0.7)

        # Case B: q_s=0.8, q_l=0.1 -> delta = 0.7 > tau2(0.05) -> fusion
        dec_b = decide(y_s, y_l, 0.8, 0.1, c)
        self.assertFalse(dec_b.accept_heavy)
        self.assertEqual(dec_b.source, "fusion")
        self.assertAlmostEqual(dec_b.delta, 0.7)

    def test_4_fusion_arithmetic(self):
        """Average and confidence-weighted fusion correctness."""
        c_avg = cfg(tau1=0.60, tau2=0.05, fusion="average")
        c_wtd = cfg(tau1=0.60, tau2=0.05, fusion="confidence_weighted")
        y_s = [10.0]
        y_l = [14.0]

        # Average fusion: delta = 0.5 - 0.3 = 0.2 > tau2(0.05) -> fusion
        dec_avg = decide(y_s, y_l, 0.5, 0.3, c_avg)
        self.assertFalse(dec_avg.accept_heavy)
        expected_avg = [(a + b) / 2 for a, b in zip(y_s, y_l)]
        np.testing.assert_allclose(dec_avg.prediction, expected_avg)

        # Confidence-weighted fusion: q_s/(q_s+q_l)*s + q_l/(q_s+q_l)*l
        q_s, q_l = 0.5, 0.3
        w_s = q_s / (q_s + q_l)
        w_l = q_l / (q_s + q_l)
        dec_wtd = decide(y_s, y_l, q_s, q_l, c_wtd)
        self.assertFalse(dec_wtd.accept_heavy)
        expected_wtd = [w_s * a + w_l * b for a, b in zip(y_s, y_l)]
        np.testing.assert_allclose(dec_wtd.prediction, expected_wtd)

        # Edge: q_s + q_l == 0 -> weight 0.5 (need tau2 < 0 so delta=0 > tau2 -> fusion)
        c_zero = cfg(tau1=0.60, tau2=-0.1, fusion="confidence_weighted")
        dec_zero = decide(y_s, y_l, 0.0, 0.0, c_zero)
        self.assertFalse(dec_zero.accept_heavy)
        expected_zero = [(a + b) / 2 for a, b in zip(y_s, y_l)]
        np.testing.assert_allclose(dec_zero.prediction, expected_zero)

    def test_5_failure_handling_propagates(self):
        """Agent failure must propagate; no silent fallback to small model."""

        class FailingBackend(MockBackend):
            def complete(self, role, payload, prompts=None, prompt_version="v1", task_instruction=""):
                raise ValueError("Agent failure")

        backend = FailingBackend()
        pipeline, _ = make_pipeline(self.config, backend=backend, gate_bias=-10.0)
        inputs = make_inputs(self.config)

        with torch.no_grad():
            with self.assertRaises(ValueError):
                pipeline(self.x, inputs, mode="sage")

    def test_6_variant_call_counts(self):
        """Verify call counts per variant with controlled gate."""
        c = cfg(tau1=0.60, tau2=0.05)

        # small_only: 0 calls
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=-10.0)
        inputs = make_inputs(c)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="small_only")
        self.assertEqual(backend.call_count, 0)

        # heavy_always: 4 calls (gate bypassed)
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=+10.0)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="heavy_always")
        self.assertEqual(backend.call_count, 4)

        # gate_only with d1=True (high confidence): 0 calls
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=+10.0)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="gate_only")
        self.assertEqual(backend.call_count, 0)

        # gate_only with d1=False (low confidence): 4 calls
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=-10.0)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="gate_only")
        self.assertEqual(backend.call_count, 4)

        # sage_average with d1=True: 0 calls
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=+10.0)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="sage")
        self.assertEqual(backend.call_count, 0)

        # sage_average with d1=False: 4 calls
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=-10.0)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="sage")
        self.assertEqual(backend.call_count, 4)

        # always_average: 4 calls regardless of gate
        backend = CountingBackend()
        pipeline, _ = make_pipeline(c, backend=backend, gate_bias=+10.0)
        with torch.no_grad():
            pipeline(self.x, inputs, mode="always_average")
        self.assertEqual(backend.call_count, 4)

    def test_no_cache_affects_call_count(self):
        """Cache hits must not produce misleading zero-call results."""
        with tempfile.TemporaryDirectory() as folder:
            backend = CountingBackend()
            pipeline, _ = make_pipeline(
                self.config, backend=backend,
                cache_dir=Path(folder) / "cache",
                gate_bias=+10.0
            )
            inputs = make_inputs(self.config)
            with torch.no_grad():
                pipeline(self.x, inputs, mode="sage")  # d1=True, no heavy call anyway
            self.assertEqual(backend.call_count, 0)

            # Now force escalation with caching enabled
            backend2 = CountingBackend()
            pipeline2, _ = make_pipeline(
                self.config, backend=backend2,
                cache_dir=Path(folder) / "cache2",
                gate_bias=-10.0
            )
            with torch.no_grad():
                pipeline2(self.x, inputs, mode="sage")  # first call: 4
                pipeline2(self.x, inputs, mode="sage")  # second call: cache hit, but backend.complete not invoked
            self.assertEqual(backend2.call_count, 4, "Cache hit should not invoke backend.complete again")


if __name__ == "__main__":
    unittest.main()
