"""Epistatic interaction mapping: site expansion and coefficient container."""

from __future__ import annotations

import itertools as it
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap

Site = tuple[int, ...]

__all__ = [
    "EpistasisMap",
    "Site",
    "assert_epistasis",
    "encoding_to_sites",
    "genotype_coeffs",
    "key_to_site",
    "site_to_key",
]


def assert_epistasis(method: Callable[..., Any]) -> Callable[..., Any]:
    """Raise AttributeError if `self.epistasis` is not set before the method runs."""

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "epistasis"):
            raise AttributeError(f"{type(self).__name__} has no .epistasis attribute set yet.")
        return method(self, *args, **kwargs)

    return wrapper


def site_to_key(site: Sequence[int], state: str = "") -> str:
    """Convert a site tuple to a comma-separated string key."""
    if not isinstance(state, str):
        raise TypeError("`state` must be a string.")
    return ",".join(str(x) for x in site) + state


def key_to_site(key: str) -> list[int]:
    """Inverse of `site_to_key` when no state suffix was used."""
    return [int(k) for k in key.split(",")]


def genotype_coeffs(genotype: str, order: int | None = None) -> list[list[int]]:
    """List all epistatic coefficients touching a binary genotype up to `order`.

    The intercept is returned as `[0]`. Mutation indices are 1-based positions
    (matching the binary-vector layout used by the design matrix).
    """
    if order is None:
        order = len(genotype)
    mutations = [i + 1 for i, c in enumerate(genotype) if c == "1"]
    coeffs: list[list[int]] = [[0]]
    for o in range(1, order + 1):
        coeffs.extend(list(z) for z in it.combinations(mutations, o))
    return coeffs


def encoding_to_sites(
    order: int,
    encoding_table: pd.DataFrame,
    start_order: int = 0,
) -> list[Site]:
    """Build the list of interaction sites up to a given order.

    Reads `site_index` and `mutation_index` from the gpmap-v2 encoding table.
    Wildtype rows (NaN `mutation_index`) are dropped.

    Parameters
    ----------
    order
        Highest interaction order to include.
    encoding_table
        Encoding table from a GenotypePhenotypeMap, one row per (site, letter).
    start_order
        If 0, the intercept term `(0,)` is prepended. If >= 1, intercept is
        omitted and only interactions of `start_order..order` are produced.
    """
    cols = encoding_table[["mutation_index", "site_index"]].dropna().astype(int)

    grouped = cols.groupby("site_index")["mutation_index"]
    per_site = [tuple(group.to_numpy()) for _, group in grouped]

    if start_order == 0:
        sites: list[Site] = [(0,)]
        orders = range(1, order + 1)
    else:
        sites = []
        orders = range(start_order, order + 1)

    for k in orders:
        for combo in it.combinations(per_site, k):
            sites.extend(it.product(*combo))

    return sites


class EpistasisMap:
    """DataFrame-backed container for epistatic coefficients.

    Columns: `labels`, `orders`, `sites`, `values`, `stdeviations`.
    """

    __slots__ = ("_data", "_gpm")

    def __init__(
        self,
        *,
        df: pd.DataFrame | None = None,
        sites: list[Site] | None = None,
        values: np.ndarray | Sequence[float] | None = None,
        stdeviations: np.ndarray | Sequence[float] | None = None,
        gpm: GenotypePhenotypeMap | None = None,
    ) -> None:
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("df must be a pandas DataFrame.")
            self._data = df.reset_index(drop=True)
        elif sites is not None:
            self._data = self._build_frame(sites, values, stdeviations)
        else:
            raise ValueError("Must supply either df= or sites=.")
        self._gpm = gpm

    @staticmethod
    def _build_frame(
        sites: list[Site],
        values: np.ndarray | Sequence[float] | None,
        stdeviations: np.ndarray | Sequence[float] | None,
    ) -> pd.DataFrame:
        n = len(sites)
        labels = [",".join(str(c) for c in s) for s in sites]
        orders = [0 if s[0] == 0 else len(s) for s in sites]
        values_arr = (
            np.asarray(values, dtype=np.float64)
            if values is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        std_arr = (
            np.asarray(stdeviations, dtype=np.float64)
            if stdeviations is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        if values_arr.shape[0] != n:
            raise ValueError(f"Expected {n} values, got {values_arr.shape[0]}.")
        if std_arr.shape[0] != n:
            raise ValueError(f"Expected {n} stdeviations, got {std_arr.shape[0]}.")
        return pd.DataFrame(
            {
                "labels": labels,
                "orders": orders,
                "sites": sites,
                "values": values_arr,
                "stdeviations": std_arr,
            }
        )

    @property
    def data(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._data)

    @property
    def n(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"EpistasisMap(n={self.n})"

    @property
    def index(self) -> pd.Index:
        return cast(pd.Index, self._data.index)

    @property
    def sites(self) -> np.ndarray:
        return np.asarray(self._data["sites"].to_numpy())

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._data["values"].to_numpy(), dtype=np.float64)

    @values.setter
    def values(self, values: np.ndarray | Sequence[float]) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] != len(self._data):
            raise ValueError(f"Expected {len(self._data)} values, got {arr.shape[0]}.")
        self._data["values"] = arr

    @property
    def stdeviations(self) -> np.ndarray:
        return np.asarray(self._data["stdeviations"].to_numpy(), dtype=np.float64)

    @stdeviations.setter
    def stdeviations(self, std: np.ndarray | Sequence[float]) -> None:
        arr = np.asarray(std, dtype=np.float64)
        if arr.shape[0] != len(self._data):
            raise ValueError(f"Expected {len(self._data)} stdeviations, got {arr.shape[0]}.")
        self._data["stdeviations"] = arr

    @property
    def gpm(self) -> GenotypePhenotypeMap:
        if self._gpm is None:
            raise AttributeError("No GenotypePhenotypeMap attached to this map.")
        return self._gpm

    def label_mapper(self) -> dict[int, str]:
        """Map each `mutation_index` to a `<W><site_label><M>` string."""
        t = self.gpm.encoding_table.dropna(subset=["mutation_index"])
        mapper: dict[int, str] = {}
        for _, row in t.iterrows():
            mapper[int(row.mutation_index)] = (
                f"{row.wildtype_letter}{row.site_label}{row.mutation_letter}"
            )
        return mapper

    @property
    def labels(self) -> list[str | list[str]]:
        mapper = self.label_mapper()
        out: list[str | list[str]] = ["w.t."]
        for term in self.sites[1:]:
            out.append([mapper[int(i)] for i in term])
        return out

    def get_orders(self, *orders: int) -> EpistasisMap:
        mask = self._data["orders"].isin(orders)
        sub = self._data.loc[mask].reset_index(drop=True)
        return EpistasisMap(df=sub, gpm=self._gpm)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> EpistasisMap:
        return cls(df=df)

    def to_dict(self) -> dict[str, list[Any]]:
        return dict(self._data.to_dict("list"))

    def to_csv(self, filename: str) -> None:
        self._data.to_csv(filename, index=False)

    def to_excel(self, filename: str) -> None:
        self._data.to_excel(filename, index=False)
