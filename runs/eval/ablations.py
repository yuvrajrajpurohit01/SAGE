"""Use `sage evaluate ... --ablations` for verify removal and fusion comparisons.

For CMAPSS narrative ablation, train a second checkpoint with --no-narrative,
then evaluate it: changing heavy inputs requires regenerating confidence labels.
No MoiraiAgent/DCATS/CoLLM reproduction results are invented by this package.
"""
from eval.benchmark import benchmark
__all__ = ["benchmark"]
