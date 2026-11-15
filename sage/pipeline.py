from dataclasses import dataclass
import time
import torch
from torch import Tensor
from sage.config import Config
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent
from sage.heavy.nexus_pipeline import ForecastInputs, HeavyPathOutputs, NexusPipeline
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder
from sage.verify.decision import decide
from sage.trace import TRACE

@dataclass
class PipelineOutputs:
    y_s: list[float]
    q_s: float
    d1: bool
    heavy: HeavyPathOutputs | None
    q_l: float | None
    delta: float | None
    d2: bool | None
    prediction: list[float]
    source: str
    elapsed_seconds: float
    mode: str

class SAGEPipeline:
    def __init__(self, config: Config, small: SmallModel, gate: FuzzyDecisionAgent,
                 heavy: NexusPipeline, reflection: SelfReflectionAgent, encoder: TraceEncoder):
        self.config, self.small, self.gate = config, small.eval(), gate.eval()
        self.heavy, self.reflection, self.encoder = heavy, reflection.eval(), encoder

    @torch.inference_mode()
    def __call__(self, x: Tensor, inputs: ForecastInputs, mode: str = "sage") -> PipelineOutputs:
        if mode not in {"sage", "small_only", "heavy_always", "gate_only", "always_average"}:
            raise ValueError("Unknown comparison mode")
        if x.shape[0] != 1 or not torch.isfinite(x).all():
            raise ValueError("Inference must receive one finite sample at a time")
        inputs.validate()
        if inputs.horizon != self.config.horizon or len(inputs.values) != self.config.window or len(inputs.feature_names) != self.config.features:
            raise ValueError("Heavy inputs must match the configured window, features and horizon")
        start = time.perf_counter()
        small = self.small(x)  # CoLLM eq. 1
        gate = self.gate(small.features)  # CoLLM eqs. 2,8,9,11; no eq. 10 labels in inference
        y_s, q_s, d1 = small.prediction[0].tolist(), gate.confidence.item(), gate.accept_small.item()
        TRACE.active = TRACE.wants(mode)
        TRACE.small(y_s)
        TRACE.gate(mode, q_s, self.config.tau1, d1)
        if mode == "small_only" or (d1 and mode not in {"heavy_always", "always_average"}):
            return PipelineOutputs(y_s, q_s, d1, None, None, None, None, y_s, "small", time.perf_counter()-start, mode)
        TRACE.heavy_start()
        heavy = self.heavy(inputs)  # SAGE replacement for CoLLM eq. 4: Nexus §§3.1–3.3
        if mode == "always_average":  # Day-1 control: heavy path on every sample, arithmetic mean, no gate/verifier
            avg = [(a + b) / 2 for a, b in zip(y_s, heavy.y_l)]
            return PipelineOutputs(y_s, q_s, d1, heavy, None, None, None, avg, "fusion", time.perf_counter()-start, mode)
        if mode in {"heavy_always", "gate_only"}:
            return PipelineOutputs(y_s, q_s, d1, heavy, None, None, None, heavy.y_l, "heavy", time.perf_counter()-start, mode)
        device = next(self.reflection.parameters()).device
        q_l = self.reflection(self.encoder(heavy).to(device)).item()  # CoLLM eqs. 5,12
        decision = decide(y_s, heavy.y_l, q_s, q_l, self.config)  # CoLLM eqs. 6,7
        TRACE.verifier(q_s, q_l, decision.delta, self.config.tau2, decision.accept_heavy, self.config.fusion)
        return PipelineOutputs(y_s, q_s, d1, heavy, q_l, decision.delta, decision.accept_heavy,
                               decision.prediction, decision.source, time.perf_counter()-start, mode)
