"""Download reproducible SAGE snapshots. API keys stay in environment variables.

Usage: python fetch_data.py --help
Data sources and limitations: DATASET_GUIDE.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import numpy as np
import pandas as pd

NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def get(url: str) -> bytes:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SAGE-Research/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        # Do not print authenticated URLs containing a query-string API key.
        raise RuntimeError(f"Provider returned HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("Provider connection failed; check network access") from None


def validate_dates(start: str, end: str) -> None:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s > e or e.date() >= datetime.now(timezone.utc).date():
        raise ValueError("Use an ordered date range ending before today")


def validate_frame(frame: pd.DataFrame, features: list[str]) -> None:
    if frame.empty or not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Provider returned empty or missing/nonfinite data; no silent imputation")
    times = pd.to_datetime(frame["timestamp"], utc=True)
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Provider timestamps must be unique and increasing")


def weather_frame(obj: dict) -> pd.DataFrame:
    if "daily" not in obj:
        raise ValueError("Weather response has no daily data")
    daily = obj["daily"]
    frame = pd.DataFrame({
        # Daily aggregates represent the COMPLETE day, not information known at midnight.
        "timestamp": pd.to_datetime(daily["time"], utc=True) + pd.Timedelta(hours=23, minutes=59, seconds=59),
        "temperature": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
        "precipitation": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"],
    })
    validate_frame(frame, ["temperature", "humidity", "precipitation", "wind_speed"])
    return frame


def stock_frame(obj: dict, start: str, end: str) -> pd.DataFrame:
    series = obj.get("Time Series (Daily)")
    if not isinstance(series, dict):
        raise ValueError("No daily series returned. Check Alpha Vantage key, quota and full-history entitlement.")
    rows = []
    for day, values in series.items():
        if start <= day <= end:
            rows.append({"timestamp": day+"T23:59:59Z", "open": float(values["1. open"]),
                         "high": float(values["2. high"]), "low": float(values["3. low"]),
                         "close": float(values["4. close"]), "volume": float(values["5. volume"])})
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True) if rows else pd.DataFrame()
    if frame.empty:
        raise ValueError("No rows in requested date interval")
    validate_frame(frame, ["open", "high", "low", "close", "volume"])
    # Full coverage is checked at coarse calendar boundaries, allowing market holidays.
    if pd.Timestamp(frame.timestamp.iloc[0]).date() > (pd.Timestamp(start)+pd.Timedelta(days=7)).date():
        raise ValueError("Stock history does not cover the requested start; full output may require a paid plan")
    if pd.Timestamp(frame.timestamp.iloc[-1]).date() < (pd.Timestamp(end)-pd.Timedelta(days=7)).date():
        raise ValueError("Stock history does not cover the requested end")
    return frame


def cmapss_files(blob: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        nested = next((n for n in outer.namelist() if n.endswith("CMAPSSData.zip")), None)
        inner_blob = outer.read(nested) if nested else blob
    result = {}
    with zipfile.ZipFile(io.BytesIO(inner_blob)) as archive:
        for name in archive.namelist():
            base = Path(name).name
            if re.fullmatch(r"(?:train|test|RUL)_FD00[1-4]\.txt", base) or base == "readme.txt":
                result[base] = archive.read(name)
    expected = {f"{role}_FD00{i}.txt" for role in ["train", "test", "RUL"] for i in range(1,5)}
    if not expected.issubset(result):
        raise ValueError("Archive missing expected CMAPSS files")
    return result


def save(folder: Path, files: dict[str, bytes], provenance: dict, overwrite: bool) -> None:
    payload = dict(files)
    provenance = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), **provenance,
        "sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in files.items()}}
    payload["provenance.json"] = json.dumps(provenance, indent=2).encode()
    if not overwrite and any((folder/name).exists() for name in payload):
        raise FileExistsError("Snapshot exists: choose a new --output or explicitly pass --overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    for name, blob in payload.items():
        (folder/name).write_bytes(blob)
    print(json.dumps({"folder": str(folder), "files": list(files), "rows": provenance.get("rows")}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="source", required=True)
    for name in ["cmapss", "weather", "stock", "ett"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", default=f"data/{'CMAPSS' if name=='cmapss' else name}")
        cmd.add_argument("--overwrite", action="store_true")
        if name in {"weather", "stock"}:
            cmd.add_argument("--start", default="2020-01-01")
            cmd.add_argument("--end", default="2025-12-31")
        if name == "weather":
            cmd.add_argument("--latitude", type=float, default=12.9165)
            cmd.add_argument("--longitude", type=float, default=79.1325)
        if name == "stock":
            cmd.add_argument("--symbol", default="IBM")
            cmd.add_argument("--demo", action="store_true", help="Use the documented IBM example key; availability is provider-controlled")
    a = p.parse_args()
    folder = Path(a.output)
    if a.source in {"weather", "stock"}:
        validate_dates(a.start, a.end)
    if a.source == "cmapss":
        blob = get(NASA_URL)
        save(folder, cmapss_files(blob), {"source": NASA_URL, "dataset": "NASA C-MAPSS",
            "archive_sha256": hashlib.sha256(blob).hexdigest()}, a.overwrite)
    elif a.source == "weather":
        params = {"latitude": a.latitude, "longitude": a.longitude, "start_date": a.start, "end_date": a.end,
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max", "timezone": "UTC", "models": "era5"}
        url = "https://archive-api.open-meteo.com/v1/archive?"+urllib.parse.urlencode(params)
        blob = get(url)
        obj = json.loads(blob)
        frame = weather_frame(obj)
        save(folder, {"weather.csv": frame.to_csv(index=False).encode(), "raw.json": blob},
            {"source": url, "rows": len(frame), "units": obj["daily_units"], "grid_latitude": obj["latitude"],
             "grid_longitude": obj["longitude"], "data_kind": "ERA5 retrospective reanalysis; NOT point-in-time observations",
             "attribution": "Open-Meteo / Copernicus Climate Change Service (ERA5)",
             "licence": "https://open-meteo.com/en/licence", "text_context": "none"}, a.overwrite)
    elif a.source == "stock":
        if a.demo and a.symbol != "IBM":
            p.error("The bundled documented demo request is for IBM only")
        key = "demo" if a.demo else os.environ.get("ALPHAVANTAGE_API_KEY", "")
        if not key:
            p.error("Set ALPHAVANTAGE_API_KEY locally, or use --demo for the IBM example")
        params = {"function": "TIME_SERIES_DAILY", "symbol": a.symbol, "outputsize": "full"}
        public_url = "https://www.alphavantage.co/query?"+urllib.parse.urlencode(params)
        blob = get(public_url+"&apikey="+urllib.parse.quote(key, safe=""))
        frame = stock_frame(json.loads(blob), a.start, a.end)
        save(folder, {"stock.csv": frame.to_csv(index=False).encode()},
            {"source": public_url, "symbol": a.symbol, "rows": len(frame), "start": a.start, "end": a.end,
             "provider_response_sha256": hashlib.sha256(blob).hexdigest(), "api_key_stored": False,
             "timestamp_semantics": "End-of-day UTC marker, not exact exchange close or data publication time",
             "prices": "TIME_SERIES_DAILY provider OHLCV; not dividend-adjusted total returns",
             "text_context": "none", "demo_request": a.demo}, a.overwrite)
    else:
        blob = get(ETT_URL)
        frame = pd.read_csv(io.BytesIO(blob)).rename(columns={"date": "timestamp"})
        validate_frame(frame, ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"])
        save(folder, {"ETTh1.csv": blob, "ett.csv": frame.to_csv(index=False).encode()},
            {"source": ETT_URL, "rows": len(frame), "target": "OT", "text_context": "none",
             "timestamp_semantics": "Source timestamps have no timezone; preserve ordering, not UTC publication semantics",
             "citation": "Zhou et al., Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, AAAI 2021",
             "dataset_page": "https://github.com/zhouhaoyi/ETDataset"}, a.overwrite)

if __name__ == "__main__":
    main()
