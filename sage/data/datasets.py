"""Three tracks; partition labels: fit / confidence / validation / test.

Only fit observations contribute to normalization. Model inference inputs
are constructed independently of supervised targets.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sage.config import Config
from sage.context.cmapss_narrative import SENSORS, narrate
from sage.heavy.nexus_pipeline import ForecastInputs, task_instruction

@dataclass
class PreparedData:
    x: np.ndarray
    raw: np.ndarray
    y: np.ndarray
    split: np.ndarray
    ids: list[str]
    timestamps: list[list[str]]
    context: list[list[str]]
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    config: Config
    target_index: int | None
    synthetic: bool = False

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def inputs(self, i: int, narrative: bool = True) -> ForecastInputs:
        context = self.context[i]
        if self.config.track.startswith("FD"):
            context = narrate(self.raw[i], self.config) if narrative else [""] * self.config.window
        return ForecastInputs(self.raw[i].tolist(), self.timestamps[i], context,
            self.feature_names, (task_instruction(self.config.prompt_version, self.config.max_rul)
            if self.target_index is None else f"Predict future {self.feature_names[self.target_index]} values in original units."),
            self.config.horizon, self.target_index)

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "arrays.npz", x=self.x, raw=self.raw, y=self.y,
            split=self.split, mean=self.mean, std=self.std)
        metadata = {key: getattr(self, key) for key in ["ids", "timestamps", "context", "feature_names", "target_index", "synthetic"]}
        metadata["config"] = asdict(self.config)
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "PreparedData":
        folder = Path(folder)
        m = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        with np.load(folder / "arrays.npz", allow_pickle=False) as a:
            arrays = {key: a[key] for key in a.files}
        return cls(**arrays, **{**m, "config": Config(**m["config"])})


def standardized(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw - mean) / std).astype(np.float32)  # CoLLM eq. 15; statistics from fit partition only


def prepare_cmapss(root: str | Path, config: Config) -> PreparedData:
    if config.track not in {"FD001", "FD003"} or config.horizon != 1:
        raise ValueError("CMAPSS supports FD001/FD003 scalar RUL")
    root = Path(root)
    train = np.loadtxt(root / f"train_{config.track}.txt")
    test = np.loadtxt(root / f"test_{config.track}.txt")
    rul = np.loadtxt(root / f"RUL_{config.track}.txt").reshape(-1)
    if train.shape[1] != 26 or test.shape[1] != 26 or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Expected finite CMAPSS rows with 26 columns")
    units = np.unique(train[:, 0]).astype(int)
    if len(units) < 5:
        raise ValueError("At least five training engines needed")
    rng = np.random.default_rng(config.seed)
    rng.shuffle(units)
    nfit, nconf = int(.6*len(units)), int(.2*len(units))
    assignments = {u: ("fit" if i < nfit else "confidence" if i < nfit+nconf else "validation") for i, u in enumerate(units)}
    cols = [sensor + 4 for sensor in SENSORS]
    fit_values = train[np.isin(train[:, 0], units[:nfit])][:, cols]
    mean, std = fit_values.mean(0), fit_values.std(0)
    std = np.where(std > 0, std, 1.)
    raw, labels, splits, ids, stamps = [], [], [], [], []
    test_units = np.unique(test[:, 0]).astype(int)
    if len(test_units) != len(rul) or not np.array_equal(test_units, np.arange(1, len(rul)+1)):
        raise ValueError("Official RUL rows must match consecutive test unit IDs")
    for table, official_test in [(train, False), (test, True)]:
        for unit in np.unique(table[:, 0]).astype(int):
            rows = table[table[:, 0] == unit]
            rows = rows[np.argsort(rows[:, 1])]
            if not np.all(np.diff(rows[:, 1]) == 1):
                raise ValueError("Engine cycles must be consecutive")
            if official_test:
                ends = [len(rows)]
            else:
                ends = range(config.window, len(rows)+1)
            for end in ends:
                observed = rows[max(0, end-config.window):end]
                # Explicit left-repeat padding for official short test histories.
                pad = config.window - len(observed)
                values = np.pad(observed[:, cols], ((pad, 0), (0, 0)), mode="edge")
                times = [f"cycle-{int(t)}" for t in observed[:, 1]]
                times = [times[0]] * pad + times
                label = rul[unit-1] if official_test else rows[-1, 1] - rows[end-1, 1]
                raw.append(values)
                labels.append([min(float(label), config.max_rul)])
                splits.append("test" if official_test else assignments[unit])
                ids.append(f"{'test' if official_test else 'train'}:{unit}:{int(rows[end-1,1])}")
                stamps.append(times)
    values = np.asarray(raw, dtype=np.float32)
    return PreparedData(standardized(values, mean, std), values, np.asarray(labels, np.float32),
        np.array(splits), ids, stamps, [[""]*config.window for _ in ids], [f"sensor_{s}" for s in SENSORS], mean, std, config, None)


def prepare_csv(path: str | Path, config: Config, target: str, features: list[str]) -> PreparedData:
    """One regularly sampled series. CSV: timestamp, features, optional context + available_at.

    Input ordering defines steps (e.g. trading sessions). Publication times must
    be <= their associated observation timestamps; revised data needs auditing.
    """
    if config.track not in {"stock", "weather", "ett"} or target not in features or len(features) != config.features:
        raise ValueError("CSV track/feature/target mismatch")
    frame = pd.read_csv(path)
    times = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Timestamps must be unique and increasing; do not shuffle time series")
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite observations must be handled explicitly")
    context = frame["context"].fillna("").astype(str).tolist() if "context" in frame else [""] * len(frame)
    if any(context):
        if "available_at" not in frame:
            raise ValueError("Text context requires available_at publication timestamps")
        available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        for i, text in enumerate(context):
            if text and (pd.isna(available.iloc[i]) or available.iloc[i] > times.iloc[i]):
                raise ValueError(f"Future or undated context in row {i}")
    n = len(frame)
    boundaries = [0, int(.5*n), int(.7*n), int(.85*n), n]
    mean, std = values[:boundaries[1]].mean(0), values[:boundaries[1]].std(0)
    std = np.where(std > 0, std, 1.)
    raw, y, split, stamps, texts, ids = [], [], [], [], [], []
    target_index = features.index(target)
    for part, lo, hi in zip(["fit", "confidence", "validation", "test"], boundaries[:-1], boundaries[1:]):
        for origin in range(max(config.window, lo), hi-config.horizon+1):
            # Forecast target intervals remain wholly inside their partition.
            raw.append(values[origin-config.window:origin])
            y.append(values[origin:origin+config.horizon, target_index])
            split.append(part)
            stamps.append([t.isoformat() for t in times.iloc[origin-config.window:origin]])
            texts.append(context[origin-config.window:origin])
            ids.append(f"origin:{times.iloc[origin-1].isoformat()}")
    if set(split) != {"fit", "confidence", "validation", "test"}:
        raise ValueError("Insufficient rows for all four partitions; reduce window or add data")
    raw = np.asarray(raw, np.float32)
    return PreparedData(standardized(raw, mean, std), raw, np.asarray(y, np.float32), np.array(split),
        ids, stamps, texts, features, mean, std, config, target_index)


def synthetic(config: Config, count: int = 96) -> PreparedData:
    """Independent generated windows for wiring tests; never CMAPSS/weather/stock data."""
    rng = np.random.default_rng(config.seed)
    raw = rng.normal(size=(count, config.window, config.features)).cumsum(axis=1).astype(np.float32)
    target_index = None if config.track.startswith("FD") else 0
    base = 100 - 3*raw.std(axis=(1,2)) if target_index is None else raw[:, -1, 0]
    y = np.stack([base + .2*i for i in range(config.horizon)], axis=1).astype(np.float32)
    splits = np.array(["fit"]*(count//2) + ["confidence"]*(count//4) + ["validation"]*(count//8) + ["test"]*(count-count//2-count//4-count//8))
    mean, std = raw[splits == "fit"].mean(axis=(0,1)), raw[splits == "fit"].std(axis=(0,1))
    ids = [f"synthetic-{i}" for i in range(count)]
    return PreparedData(standardized(raw, mean, std), raw, y, splits, ids,
        [[f"step-{t}" for t in range(config.window)] for _ in ids], [[""]*config.window for _ in ids],
        [f"sensor_{s}" for s in SENSORS] if target_index is None else [f"value_{i}" for i in range(config.features)],
        mean, std, config, target_index, True)
