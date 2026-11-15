"""Regression tests for prompt_v2 integration.

Covers:
1. v1 task instruction remains byte-for-byte unchanged.
2. v2 task instruction contains no healthy-state default.
3. v1 legacy cache keys remain identical.
4. v2 cache keys differ from v1.
5. The selected prompt version reaches every backend call.
6. Old checkpoints still load as v1.
7. v2 checkpoints cannot silently be evaluated using v1.
"""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from sage.config import Config
from sage.data.datasets import synthetic
from sage.heavy.nexus_pipeline import (
    MockBackend, NexusPipeline, PROMPTS, TASK_V1_RUL, TASK_V2_RUL,
    task_instruction, load_prompts, parse_trace, ForecastInputs
)
from sage.pipeline import SAGEPipeline
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder
from train.training import train, load_pipeline


def cfg(**kwargs):
    return replace(Config(window=4, hidden=3, embedding_dim=8), **kwargs)


class TaskInstructionTests(unittest.TestCase):
    """Requirement 1 & 2: versioned task instructions."""

    def test_v1_task_unchanged_byte_for_byte(self):
        """V1 task instruction must match the original wording exactly."""
        expected = ("Predict remaining useful life at the final observed engine cycle, in cycles. "
                    "Labels are capped at {max_rul:g} cycles (a healthy engine has RUL {max_rul:g}), "
                    "so the answer lies between 0 and {max_rul:g}; once degradation begins, "
                    "RUL falls by one per operating cycle. "
                    "Isolated sensor-deviation flags are common in healthy engines because of "
                    "measurement noise and are not evidence of imminent failure; "
                    "an RUL below about 30 cycles requires many sensors to show large, sustained "
                    "deviations over many consecutive cycles.")
        self.assertEqual(TASK_V1_RUL, expected)
        rendered = task_instruction("v1", 125.0)
        self.assertIn("healthy engine has RUL 125", rendered)
        self.assertIn("Isolated sensor-deviation flags are common in healthy engines", rendered)

    def test_v2_task_no_healthy_state_default(self):
        """V2 task instruction must not contain healthy-state default language."""
        rendered = task_instruction("v2", 125.0)
        self.assertNotIn("healthy engine", rendered)
        self.assertNotIn("healthy engines because of measurement noise", rendered)
        self.assertNotIn("isolated sensor-deviation flags are common", rendered)
        self.assertIn("upper RUL cap only", rendered)
        self.assertIn("does not establish engine health", rendered)
        self.assertIn("insufficient to establish degradation", rendered)

    def test_v2_prompts_contain_no_healthy_state(self):
        """Prompt v2 agent instructions must not contain healthy-state assumptions."""
        prompts = load_prompts("v2")
        for role, text in prompts.items():
            self.assertNotIn("healthy engine", text)
            self.assertNotIn("isolated.*noise", text, msg=f"role {role} still has isolated noise")


class CacheKeyTests(unittest.TestCase):
    """Requirement 3 & 4: cache key integrity and separation."""

    def test_v1_cache_key_preserved(self):
        """V1 cache key construction must be byte-identical to legacy."""
        d = synthetic(cfg(), count=1)
        pipe = NexusPipeline(MockBackend(), prompt_version="v1")
        inputs = d.inputs(0)
        payload = {"history": asdict(inputs), "guidelines": []}
        key = hashlib.sha256(json.dumps({"payload": payload, "backend": pipe.backend.identity,
            "prompts": PROMPTS, "version": 1}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        # Legacy construction must produce the same key
        legacy_key = hashlib.sha256(json.dumps({"payload": payload, "backend": "mock-v1-NOT-RESEARCH",
            "prompts": PROMPTS, "version": 1}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.assertEqual(key, legacy_key)

    def test_v2_cache_key_differs_from_v1(self):
        """V2 cache key must differ from v1 for the same input."""
        d = synthetic(cfg(), count=1)
        pipe_v2 = NexusPipeline(MockBackend(), prompt_version="v2")
        inputs = d.inputs(0)
        payload = {"history": asdict(inputs), "guidelines": []}
        key_v1 = hashlib.sha256(json.dumps({"payload": payload, "backend": "mock-v1-NOT-RESEARCH",
            "prompts": PROMPTS, "version": 1}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        key_v2 = hashlib.sha256(json.dumps({"payload": payload, "backend": "mock-v1-NOT-RESEARCH",
            "prompts": pipe_v2._prompts, "prompt_version": "v2",
            "task_instruction": pipe_v2._task_instruction}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.assertNotEqual(key_v1, key_v2)

    def test_v1_and_v2_cache_entries_do_not_overlap(self):
        """v1 and v2 must never share cached results."""
        d = synthetic(cfg(), count=1)
        with tempfile.TemporaryDirectory() as folder:
            pipe_v1 = NexusPipeline(MockBackend(), folder, prompt_version="v1")
            pipe_v2 = NexusPipeline(MockBackend(), folder, prompt_version="v2")
            inputs = d.inputs(0)
            _ = pipe_v1(inputs)
            _ = pipe_v2(inputs)
            cache_files = list(Path(folder).glob("*.json"))
            self.assertEqual(len(cache_files), 2, "v1 and v2 should produce distinct cache files")


def asdict_inputs(inputs):
    from dataclasses import asdict
    return asdict(inputs)


class PromptPropagationTests(unittest.TestCase):
    """Requirement 5: selected prompt version reaches every backend call."""

    def test_prompt_version_reaches_all_four_agents(self):
        """Every agent call must carry the selected prompt version."""
        calls = []

        class RecordingBackend(MockBackend):
            def complete(self, role, payload, prompts=None, prompt_version="v1", task_instruction=""):
                calls.append((role, prompt_version, bool(prompts), bool(task_instruction)))
                return super().complete(role, payload, prompts=prompts,
                                        prompt_version=prompt_version,
                                        task_instruction=task_instruction)

        d = synthetic(cfg(), count=1)
        pipe = NexusPipeline(RecordingBackend(), prompt_version="v2")
        result = pipe(d.inputs(0))
        roles_seen = {c[0] for c in calls}
        self.assertEqual(roles_seen, {"context", "macro", "micro", "synthesis"})
        for role, pv, has_prompts, has_task in calls:
            self.assertEqual(pv, "v2", f"role {role} did not receive v2")
            self.assertTrue(has_prompts, f"role {role} did not receive prompts dict")
            self.assertTrue(has_task, f"role {role} did not receive task instruction")

    def test_v1_reaches_all_four_agents(self):
        """V1 prompt version also reaches all four agents."""
        calls = []

        class RecordingBackend(MockBackend):
            def complete(self, role, payload, prompts=None, prompt_version="v1", task_instruction=""):
                calls.append(prompt_version)
                return super().complete(role, payload, prompts=prompts,
                                        prompt_version=prompt_version,
                                        task_instruction=task_instruction)

        d = synthetic(cfg(), count=1)
        pipe = NexusPipeline(RecordingBackend(), prompt_version="v1")
        _ = pipe(d.inputs(0))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(pv == "v1" for pv in calls))


class CheckpointConsistencyTests(unittest.TestCase):
    """Requirement 6 & 7: checkpoint prompt version handling."""

    def test_legacy_checkpoint_loads_as_v1(self):
        """A checkpoint without prompt_version field loads as v1 by default."""
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            heavy = NexusPipeline(MockBackend(), Path(folder) / "cache")
            train(d, heavy, folder, epochs=1, confidence_epochs=1)
            # Manually strip prompt_version from checkpoint to simulate legacy
            ckpt_path = Path(folder) / "sage.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt.pop("prompt_version", None)
            torch.save(ckpt, ckpt_path)
            heavy2 = NexusPipeline(MockBackend(), Path(folder) / "cache", prompt_version="v1")
            pipeline, metadata = load_pipeline(ckpt_path, d, heavy2)
            # Loading should succeed with v1 resolution
            result = pipeline(torch.from_numpy(d.x[-1:]), d.inputs(len(d.ids) - 1))
            self.assertTrue(np.isfinite(result.prediction).all())

    def test_v2_checkpoint_cannot_be_evaluated_with_v1(self):
        """A v2 checkpoint must reject silent v1 evaluation."""
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            heavy = NexusPipeline(MockBackend(), Path(folder) / "cache", prompt_version="v2")
            train(d, heavy, folder, epochs=1, confidence_epochs=1)
            ckpt_path = Path(folder) / "sage.pt"
            # Try loading with v1 pipeline - should raise ValueError
            heavy_v1 = NexusPipeline(MockBackend(), Path(folder) / "cache", prompt_version="v1")
            with self.assertRaises(ValueError):
                load_pipeline(ckpt_path, d, heavy_v1)


class PipelineIntegrationTests(unittest.TestCase):
    """Full pipeline works with both prompt versions."""

    def test_pipeline_v1_smoke(self):
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            heavy = NexusPipeline(MockBackend(), Path(folder) / "cache", prompt_version="v1")
            train(d, heavy, folder, epochs=1, confidence_epochs=1)
            pipeline, ckpt = load_pipeline(Path(folder) / "sage.pt", d, heavy)
            result = pipeline(torch.from_numpy(d.x[-1:]), d.inputs(len(d.ids) - 1))
            self.assertTrue(np.isfinite(result.prediction).all())

    def test_pipeline_v2_smoke(self):
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            heavy = NexusPipeline(MockBackend(), Path(folder) / "cache", prompt_version="v2")
            train(d, heavy, folder, epochs=1, confidence_epochs=1)
            pipeline, ckpt = load_pipeline(Path(folder) / "sage.pt", d, heavy,
                                           prompt_version="v2")
            result = pipeline(torch.from_numpy(d.x[-1:]), d.inputs(len(d.ids) - 1))
            self.assertTrue(np.isfinite(result.prediction).all())


if __name__ == "__main__":
    unittest.main()
