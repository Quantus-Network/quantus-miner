#!/usr/bin/env python3
"""Fit hashrate ~ sm_count (+ optional vram_mb) from results.csv.

Economics (ideal_hash_per_dollar) are not modeled from silicon alone —
use predicted_hashrate / cost_per_sec with an offer's price instead.

Usage:
  ./fit-hashrate.py
  ./fit-hashrate.py --csv results.csv --min-util 50
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def fnum(x: str | None) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def solve_least_squares(X: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares via normal equations (no numpy)."""
    n = len(X)
    p = len(X[0])
    # XtX and Xty
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for i in range(n):
        for a in range(p):
            xty[a] += X[i][a] * y[i]
            for b in range(p):
                xtx[a][b] += X[i][a] * X[i][b]
    # Gaussian elimination
    m = [xtx[r][:] + [xty[r]] for r in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise RuntimeError("singular design matrix")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, p + 1):
            m[col][j] /= div
        for r in range(p):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, p + 1):
                m[r][j] -= factor * m[col][j]
    return [m[r][p] for r in range(p)]


def r2_score(y: list[float], yhat: list[float]) -> float:
    mean = sum(y) / len(y)
    ss_tot = sum((yi - mean) ** 2 for yi in y)
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results.csv",
    )
    ap.add_argument("--min-util", type=float, default=50.0)
    ap.add_argument(
        "--no-vram",
        action="store_true",
        help="fit hashrate ~ sm_count only",
    )
    args = ap.parse_args()

    rows_in = list(csv.DictReader(args.csv.open(newline="")))
    samples: list[tuple[str, float, float, float | None, float]] = []
    for r in rows_in:
        util = fnum(r.get("gpu_utilization_pct"))
        hs = fnum(r.get("hashrate"))
        sm = fnum(r.get("sm_count"))
        vram = fnum(r.get("vram_mb"))
        if util is None or util < args.min_util or hs is None or sm is None:
            continue
        samples.append((r.get("gpu_model", "?"), hs, sm, vram, util))

    # One fit point per GPU model: keep the best-H/s row so repeated sweeps and
    # deliberately suboptimal batch-size rows don't overweight a model.
    best: dict[str, tuple[str, float, float, float | None, float]] = {}
    for s in samples:
        if s[0] not in best or s[1] > best[s[0]][1]:
            best[s[0]] = s
    samples = list(best.values())

    if len(samples) < 3:
        raise SystemExit(f"need ≥3 rows with util≥{args.min_util} and sm_count; got {len(samples)}")

    use_vram = not args.no_vram and all(s[3] is not None for s in samples)
    y = [s[1] for s in samples]
    if use_vram:
        X = [[1.0, s[2], float(s[3])] for s in samples]
        names = ["intercept", "sm_count", "vram_mb"]
    else:
        X = [[1.0, s[2]] for s in samples]
        names = ["intercept", "sm_count"]

    beta = solve_least_squares(X, y)
    yhat = [sum(b * x for b, x in zip(beta, xi)) for xi in X]
    r2 = r2_score(y, yhat)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, yhat)) / len(y))

    print(f"n={len(samples)}  min_util={args.min_util}  features={names}")
    print("hashrate ≈ " + " + ".join(f"{b:.4g}*{n}" for b, n in zip(beta, names)))
    print(f"R²={r2:.4f}  RMSE={rmse/1e6:.3f} MH/s")
    print()
    print("Per-GPU residual (best H/s row per model, sorted by |err|):")
    ranked = sorted(
        zip(samples, yhat),
        key=lambda t: abs(t[0][1] - t[1]),
        reverse=True,
    )
    for (gpu, hs, sm, vram, util), pred in ranked[:15]:
        err = hs - pred
        print(
            f"  {err/1e6:+6.2f}M err  actual={hs/1e6:6.2f}M  pred={pred/1e6:6.2f}M  "
            f"sm={sm:.0f}  vram={vram or float('nan'):.0f}  util={util:.1f}%  {gpu}"
        )
    print()
    print("To score an offer: predicted_H/s from the formula, then")
    print("  predicted_ideal_H/$ ≈ predicted_H/s / (cost_per_hour/3600)")


if __name__ == "__main__":
    main()
