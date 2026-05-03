"""Benchmark epistasis-v2 fit() times.

This script benchmarks the installed epistasis package (v2). To reproduce the
v1 vs v2 comparison from the README, run it twice in separate environments:

    # v1 environment
    uv venv .venv-v1 --python 3.11
    uv pip install --python .venv-v1/Scripts/python.exe \
        numpy scipy scikit-learn pandas lmfit matplotlib emcee gpmap==0.7.0
    # copy epistasis 0.7.5 pure-Python source into .venv-v1/Lib/site-packages/epistasis
    .venv-v1/Scripts/python vs_v1.py

    # v2 environment (from the repo root)
    uv run python benchmarks/vs_v1.py

Operations timed:
  fit_order1  -- order-1 linear regression (sklearn lstsq)
  fit_full    -- full-order regression (v2: FWHT O(N log N); v1: dense lstsq)
"""

from __future__ import annotations

import itertools
import json
import timeit
from pathlib import Path

import numpy as np

SIZES_ORDER1 = [8, 10, 12, 14, 16]
SIZES_FULL = [8, 10, 12, 14, 16]
N_REPEATS = 5


def make_gpm(L: int):
    from gpmap import GenotypePhenotypeMap

    genotypes = ["".join(g) for g in itertools.product("AT", repeat=L)]
    phenotypes = np.random.default_rng(42).normal(size=len(genotypes))
    stdeviations = np.full(len(genotypes), 0.1)
    return GenotypePhenotypeMap(
        wildtype="A" * L,
        genotypes=genotypes,
        phenotypes=phenotypes,
        stdeviations=stdeviations,
    )


def best_ms(fn, n: int) -> float:
    return min(timeit.repeat(fn, number=1, repeat=n)) * 1000


def fit(gpm, order: int) -> None:
    try:
        from epistasis.models.linear import EpistasisLinearRegression  # v2
    except ImportError:
        from epistasis.models import EpistasisLinearRegression  # v1 fallback

    m = EpistasisLinearRegression(order=order)
    m.add_gpm(gpm)
    m.fit()


try:
    from epistasis import __version__ as _ver
except Exception:
    _ver = "unknown"

results: dict = {"version": _ver, "results": {}}

print(f"epistasis {_ver}")

print("fit() order=1")
fit_order1: dict[str, float] = {}
for L in SIZES_ORDER1:
    gpm = make_gpm(L)
    t = best_ms(lambda g=gpm: fit(g, 1), N_REPEATS)
    fit_order1[f"L{L}"] = round(t, 4)
    print(f"  L={L:2d} ({2**L:6d} genotypes): {t:.3f} ms")
results["results"]["fit_order1_ms"] = fit_order1

print("fit() full order")
fit_full: dict[str, float] = {}
for L in SIZES_FULL:
    gpm = make_gpm(L)
    t = best_ms(lambda g=gpm, k=L: fit(g, k), N_REPEATS)
    fit_full[f"L{L}"] = round(t, 4)
    print(f"  L={L:2d} ({2**L:6d} genotypes): {t:.3f} ms")
results["results"]["fit_full_ms"] = fit_full

out = Path(__file__).parent / f"results_{_ver.replace('.', '_')}.json"
out.write_text(json.dumps(results, indent=2))
print(f"Saved {out}")
