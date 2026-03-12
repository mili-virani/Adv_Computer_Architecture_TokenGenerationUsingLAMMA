import csv
import os
import statistics
from typing import List, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    idx = int(round((p / 100.0) * (len(v) - 1)))
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])

def trimmed_mean(vals: List[float], trim_ratio: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = int(len(v) * trim_ratio)
    if len(v) <= 2 * k:
        return float(sum(v) / len(v))
    v2 = v[k:len(v)-k]
    return float(sum(v2) / len(v2))

def iqr_filter(vals: List[float], k: float = 1.5) -> List[float]:
    """IQR-based outlier removal. Returns filtered list."""
    if len(vals) < 4:
        return vals[:]
    v = sorted(vals)
    q1 = v[len(v)//4]
    q3 = v[(3*len(v))//4]
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return [x for x in vals if lo <= x <= hi]

def stats_dict(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }

def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def safe_name(model_id: str) -> str:
    return model_id.replace("/", "_")