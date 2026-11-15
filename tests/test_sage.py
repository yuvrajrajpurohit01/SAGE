import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from sage.config import Config
from sage.data.datasets import synthetic, prepare_csv, prepare_cmapss, PreparedData
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent, confidence_target
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder
from sage.verify.decision import decide
from sage.heavy.nexus_pipeline import MockBackend, NexusPipeline, parse_trace
from sage.pipeline import SAGEPipeline
from sage.context.cmapss_narrative import narrate
from train.training import train, load_pipeline
from eval.benchmark import metrics

def cfg(**kwargs):
    return replace(Config(window=4, hidden=3, embedding_dim=8), **kwargs)

class MathTests(unittest.TestCase):
    def test_membership_exact_no_factor_two(self):
        c = cfg()
        gate = FuzzyDecisionAgent(c)
        z = torch.ones(2, c.window, c.hidden)
        result = gate(z)
        torch.testing.assert_close(result.membership, torch.full_like(z, np.exp(-1)))
        result.confidence.sum().backward()
        self.assertIsNotNone(gate.mu.grad)
        self.assertIsNotNone(gate.log_variance.grad)

    def test_confidence_targets_detach_and_scalar_formula(self):
        p = torch.tensor([[0.], [2.]], requires_grad=True)
        y = torch.zeros_like(p)
        expected = torch.tensor([[1.], [1-np.tanh(1.)]], dtype=torch.float32)
        for target in [confidence_target(p, y, 2), SelfReflectionAgent.target(p, y, 2)]:
            torch.testing.assert_close(target, expected)
            self.assertFalse(target.requires_grad)
        self.assertAlmostEqual(SelfReflectionAgent.loss(torch.tensor([[.5],[.5]]), torch.tensor([[1.],[0.]])).item(), .25)

    def test_multistep_extension_is_mean_of_labels(self):
        target = confidence_target(torch.tensor([[0., 2.]]), torch.zeros(1,2), 2.)
        self.assertAlmostEqual(target.item(), (2-np.tanh(1))/2, places=6)

    def test_decision_equality_and_fusion(self):
        c = cfg(tau2=.25)
        self.assertEqual(decide([0.], [10.], .5, .25, c).source, "heavy")
        self.assertEqual(decide([0.], [10.], .6, .25, c).prediction, [5.])
        self.assertAlmostEqual(decide([0.], [10.], .75, .25, replace(c, fusion="confidence_weighted")).prediction[0], 2.5)
        with self.assertRaises(ValueError):
            decide([0.], [float("nan")], .5, .3, c)

    def test_metrics_and_zero_mape(self):
        m = metrics(np.array([[0.], [2.]]), np.array([[1.], [3.]]))
        self.assertEqual(m["rmse"], 1.)
        self.assertEqual(m["mae"], 1.)
        self.assertEqual(m["mape_excluded_near_zero"], 1)
        self.assertEqual(m["mape_percent_nonzero_targets"], 50.)

class PipelineTests(unittest.TestCase):
    def make(self, c, heavy):
        small, gate, reflection = SmallModel(c), FuzzyDecisionAgent(c), SelfReflectionAgent(c)
        with torch.no_grad():
            gate.head.weight.zero_()
            gate.head.bias.zero_()
            reflection.projection.weight.zero_()
            reflection.projection.bias.zero_()
        return SAGEPipeline(c, small, gate, heavy, reflection, TraceEncoder(c))

    def test_fast_exit_never_calls_heavy(self):
        class Bomb:
            def __call__(self, inputs):
                raise AssertionError("Heavy path was called")
        c = cfg(tau1=.5)
        d = synthetic(c)
        result = self.make(c, Bomb())(torch.from_numpy(d.x[:1]), d.inputs(0))
        self.assertEqual(result.source, "small")
        self.assertIsNone(result.heavy)
        self.assertIsNone(result.q_l)

    def test_heavy_accept_and_fusion_routes(self):
        c = cfg(tau1=.6)
        d = synthetic(c)
        p = self.make(c, NexusPipeline(MockBackend()))
        x, inputs = torch.from_numpy(d.x[:1]), d.inputs(0)
        self.assertEqual(p(x, inputs).source, "heavy")
        with torch.no_grad():
            p.reflection.projection.bias.fill_(-4.)
        fused = p(x, inputs)
        self.assertEqual(fused.source, "fusion")
        np.testing.assert_allclose(fused.prediction, (np.array(fused.y_s)+np.array(fused.heavy.y_l))/2)

    def test_independent_outlooks_cache_and_schema_validation(self):
        class Recorder(MockBackend):
            def __init__(self): self.calls = []
            def complete(self, role, payload, prompts=None, prompt_version="v1", task_instruction=""):
                self.calls.append((role, copy.deepcopy(payload)))
                return super().complete(role, payload, prompts=prompts,
                                        prompt_version=prompt_version,
                                        task_instruction=task_instruction)
        d = synthetic(cfg())
        with tempfile.TemporaryDirectory() as folder:
            b = Recorder()
            p = NexusPipeline(b, folder)
            first, second = p(d.inputs(0)), p(d.inputs(0))
            self.assertEqual(len(b.calls), 4)
            self.assertNotIn("macro", b.calls[2][1])
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.y_l, second.y_l)
            for _, payload in b.calls:
                self.assertNotIn("truth", json.dumps(payload))
            with self.assertRaises(ValueError):
                parse_trace("micro", {"prediction": [3.], "reasoning": "ok", "step_reasoning": []}, d.inputs(0), (0,0))
            with self.assertRaises(ValueError):
                parse_trace("macro", {"prediction": ["3"], "reasoning": "ok"}, d.inputs(0), (0,0))

    def test_train_reload_and_backend_guard(self):
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            heavy = NexusPipeline(MockBackend(), Path(folder)/"cache")
            train(d, heavy, folder, epochs=1, confidence_epochs=1)
            p, metadata = load_pipeline(Path(folder)/"sage.pt", d, heavy)
            self.assertTrue(metadata["trained"])
            self.assertEqual(len(metadata["confidence_ids"]), 8)
            result = p(torch.from_numpy(d.x[-1:]), d.inputs(len(d.ids)-1))
            self.assertTrue(np.isfinite(result.prediction).all())
            bad = MockBackend()
            bad.identity = "different-backend"
            with self.assertRaises(ValueError):
                load_pipeline(Path(folder)/"sage.pt", d, NexusPipeline(bad))

class ExperimentTests(unittest.TestCase):
    def test_unseen_inference_and_label_rejection(self):
        from sage.inference import predict_file
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            heavy = NexusPipeline(MockBackend(), root/"cache")
            train(d, heavy, root, epochs=1, confidence_epochs=1)
            inp = d.inputs(0)
            obj = {"values": inp.values, "timestamps": inp.timestamps, "feature_names": inp.feature_names}
            path = root/"history.json"
            path.write_text(json.dumps(obj))
            result = predict_file(root/"sage.pt", path, heavy)
            self.assertEqual(len(result["prediction"]), 1)
            obj["truth"] = [2.]
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                predict_file(root/"sage.pt", path, heavy)

    def test_threshold_selection_only_uses_validation_ids(self):
        from eval.tune import tune
        d = synthetic(cfg(), 32)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            heavy = NexusPipeline(MockBackend(), root/"cache")
            train(d, heavy, root, epochs=1, confidence_epochs=1)
            pipeline, ckpt = load_pipeline(root/"sage.pt", d, heavy)
            tune(pipeline, d, ckpt, root/"tuned", [.3,.6], [.05,.1])
            tuned = torch.load(root/"tuned"/"sage-tuned.pt", weights_only=True)
            selected = set(tuned["threshold_selection"]["sample_ids"])
            self.assertEqual(selected, {d.ids[i] for i in d.indices("validation")})
            self.assertFalse(selected & {d.ids[i] for i in d.indices("test")})

class DataTests(unittest.TestCase):
    def test_narrative_is_causal(self):
        c = cfg()
        x = np.random.default_rng(3).normal(size=(10,14))
        first = narrate(x, c)
        x[6:] = 10000
        self.assertEqual(first[:6], narrate(x, c)[:6])

    def test_csv_split_normalization_and_context_availability(self):
        c = cfg(track="weather", features=1, horizon=3)
        dates = pd.date_range("2020-01-01", periods=100, tz="UTC")
        frame = pd.DataFrame({"timestamp": dates, "temperature": np.arange(100, dtype=float),
                              "context": "observation", "available_at": dates})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weather.csv"
            frame.to_csv(path, index=False)
            d = prepare_csv(path, c, "temperature", ["temperature"])
            self.assertEqual(d.mean[0], 24.5)
            self.assertLess(d.y[d.indices("fit")].max(), 50)
            self.assertGreaterEqual(d.y[d.indices("confidence")].min(), 50)
            self.assertLess(d.y[d.indices("confidence")].max(), 70)
            d.save(Path(folder)/"prepared")
            reread = PreparedData.load(Path(folder)/"prepared")
            np.testing.assert_array_equal(d.x, reread.x)
            frame.loc[90, "temperature"] = 1e10
            frame.to_csv(path, index=False)
            newer = prepare_csv(path, c, "temperature", ["temperature"])
            np.testing.assert_array_equal(d.x[d.indices("fit")], newer.x[newer.indices("fit")])
            frame.loc[10, "available_at"] = dates[11]
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                prepare_csv(path, c, "temperature", ["temperature"])

    def test_cmapss_units_and_short_test_padding(self):
        c = cfg()
        def table(units, cycles):
            return np.array([[u, t, 0,0,0, *[u+t+j for j in range(21)]]
                             for u in range(1,units+1) for t in range(1,cycles+1)], float)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savetxt(root/"train_FD001.txt", table(10,6))
            np.savetxt(root/"test_FD001.txt", table(2,2))
            np.savetxt(root/"RUL_FD001.txt", [30,40])
            d = prepare_cmapss(root, c)
            partitions = [{d.ids[i].split(':')[1] for i in d.indices(s)} for s in ["fit", "confidence", "validation"]]
            self.assertFalse(partitions[0] & partitions[1])
            self.assertFalse(partitions[1] & partitions[2])
            self.assertEqual(d.raw.shape[2], 14)
            self.assertEqual(len(d.indices("test")), 2)
            np.testing.assert_array_equal(d.y[d.indices("test")].ravel(), [30,40])
            i = d.indices("test")[0]
            np.testing.assert_array_equal(d.raw[i,0], d.raw[i,1])

if __name__ == "__main__":
    unittest.main()
