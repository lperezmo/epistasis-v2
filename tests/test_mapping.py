"""Tests for epistasis.mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from epistasis.mapping import (
    EpistasisMap,
    assert_epistasis,
    encoding_to_sites,
    genotype_coeffs,
    key_to_site,
    site_to_key,
)
from gpmap import GenotypePhenotypeMap


def _biallelic_encoding_table() -> pd.DataFrame:
    """Minimal encoding table for a 2-site biallelic system (00/01/10/11)."""
    return pd.DataFrame(
        {
            "site_index": [0, 0, 1, 1],
            "wildtype_letter": ["A", "A", "A", "A"],
            "mutation_letter": ["A", "B", "A", "B"],
            "mutation_index": [pd.NA, 1, pd.NA, 2],
        }
    )


def _ternary_encoding_table() -> pd.DataFrame:
    """Encoding table for a 2-site ternary system (A/B/C at each site)."""
    return pd.DataFrame(
        {
            "site_index": [0, 0, 0, 1, 1, 1],
            "wildtype_letter": ["A"] * 6,
            "mutation_letter": ["A", "B", "C", "A", "B", "C"],
            "mutation_index": [pd.NA, 1, 2, pd.NA, 3, 4],
        }
    )


# site_to_key / key_to_site.


def test_site_to_key_basic() -> None:
    assert site_to_key((1, 2, 3)) == "1,2,3"
    assert site_to_key((0,)) == "0"


def test_site_to_key_with_state() -> None:
    assert site_to_key((1, 2), state="+") == "1,2+"


def test_site_to_key_rejects_non_string_state() -> None:
    with pytest.raises(TypeError):
        site_to_key((1, 2), state=42)  # type: ignore[arg-type]


def test_key_to_site_roundtrip() -> None:
    site = (1, 4, 7)
    assert tuple(key_to_site(site_to_key(site))) == site


# genotype_coeffs.


def test_genotype_coeffs_wildtype() -> None:
    assert genotype_coeffs("000") == [[0]]


def test_genotype_coeffs_single_mutation() -> None:
    assert genotype_coeffs("100") == [[0], [1]]


def test_genotype_coeffs_all_orders() -> None:
    out = genotype_coeffs("111")
    assert out == [[0], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]


def test_genotype_coeffs_order_cap() -> None:
    out = genotype_coeffs("111", order=2)
    assert out == [[0], [1], [2], [3], [1, 2], [1, 3], [2, 3]]


# encoding_to_sites.


def test_encoding_to_sites_biallelic_order2() -> None:
    sites = encoding_to_sites(order=2, encoding_table=_biallelic_encoding_table())
    assert sites == [(0,), (1,), (2,), (1, 2)]


def test_encoding_to_sites_biallelic_order1_has_intercept() -> None:
    sites = encoding_to_sites(order=1, encoding_table=_biallelic_encoding_table())
    assert sites == [(0,), (1,), (2,)]


def test_encoding_to_sites_start_order_skips_intercept() -> None:
    sites = encoding_to_sites(order=2, encoding_table=_biallelic_encoding_table(), start_order=1)
    assert (0,) not in sites
    assert sites == [(1,), (2,), (1, 2)]


def test_encoding_to_sites_ternary_order2() -> None:
    sites = encoding_to_sites(order=2, encoding_table=_ternary_encoding_table())
    assert (0,) in sites
    assert (1,) in sites and (2,) in sites and (3,) in sites and (4,) in sites
    assert (1, 3) in sites and (2, 4) in sites
    # 1 intercept + 4 first-order + 4 pairwise products across 2 sites
    assert len(sites) == 9


def test_encoding_to_sites_drops_wildtype_rows() -> None:
    """Rows with NaN mutation_index should not contribute sites."""
    t = _biallelic_encoding_table()
    sites = encoding_to_sites(order=2, encoding_table=t)
    # No site should include index 0 except the intercept.
    nonzero_sites = [s for s in sites if s != (0,)]
    for s in nonzero_sites:
        assert 0 not in s


# encoding_to_sites against a real gpmap.


def test_encoding_to_sites_with_gpmap() -> None:
    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
    assert sites[0] == (0,)
    assert len(sites) == 4


# EpistasisMap.


def test_epistasis_map_from_sites_default_values_are_nan() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,), (1, 2)])
    assert em.n == 4
    assert len(em) == 4
    assert np.isnan(em.values).all()
    assert np.isnan(em.stdeviations).all()


def test_epistasis_map_from_sites_with_values() -> None:
    em = EpistasisMap(
        sites=[(0,), (1,), (2,), (1, 2)],
        values=[1.0, 0.5, -0.3, 0.1],
    )
    np.testing.assert_allclose(em.values, [1.0, 0.5, -0.3, 0.1])


def test_epistasis_map_labels_column() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,), (1, 2)])
    assert list(em.data["labels"]) == ["0", "1", "2", "1,2"]


def test_epistasis_map_orders_column() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,), (1, 2)])
    assert list(em.data["orders"]) == [0, 1, 1, 2]


def test_epistasis_map_values_setter_validates_length() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,)])
    with pytest.raises(ValueError):
        em.values = [1.0, 2.0]


def test_epistasis_map_values_setter_roundtrip() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,)])
    em.values = [0.5, -0.1, 0.9]
    np.testing.assert_allclose(em.values, [0.5, -0.1, 0.9])


def test_epistasis_map_stdeviations_setter_validates_length() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,)])
    with pytest.raises(ValueError):
        em.stdeviations = [0.1, 0.2]


def test_epistasis_map_from_dataframe_roundtrip() -> None:
    original = EpistasisMap(
        sites=[(0,), (1,), (2,), (1, 2)],
        values=[1.0, 0.5, -0.3, 0.1],
    )
    restored = EpistasisMap.from_dataframe(original.data)
    np.testing.assert_allclose(restored.values, original.values)
    assert list(restored.data["labels"]) == list(original.data["labels"])


def test_epistasis_map_get_orders() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,), (3,), (1, 2), (1, 3), (1, 2, 3)])
    only_pairs = em.get_orders(2)
    assert only_pairs.n == 2
    assert list(only_pairs.data["orders"]) == [2, 2]


def test_epistasis_map_get_orders_multiple() -> None:
    em = EpistasisMap(sites=[(0,), (1,), (2,), (1, 2), (1, 2, 3)])
    subset = em.get_orders(0, 2)
    assert list(subset.data["orders"]) == [0, 2]


def test_epistasis_map_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        EpistasisMap(df="not a dataframe")  # type: ignore[arg-type]


def test_epistasis_map_requires_df_or_sites() -> None:
    with pytest.raises(ValueError):
        EpistasisMap()


def test_epistasis_map_gpm_not_set() -> None:
    em = EpistasisMap(sites=[(0,), (1,)])
    with pytest.raises(AttributeError):
        _ = em.gpm


def test_epistasis_map_values_dtype_is_float64() -> None:
    em = EpistasisMap(sites=[(0,), (1,)], values=[1, 2])
    assert em.values.dtype == np.float64


def test_epistasis_map_to_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    em = EpistasisMap(sites=[(0,), (1,), (1, 2)], values=[1.0, 0.5, 0.1])
    path = tmp_path / "coefs.csv"
    em.to_csv(str(path))
    loaded = pd.read_csv(path)
    np.testing.assert_allclose(loaded["values"].to_numpy(), [1.0, 0.5, 0.1])


# Integration with gpmap: labels + label_mapper.


def test_epistasis_map_label_mapper_with_gpm() -> None:
    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
    em = EpistasisMap(sites=sites, gpm=gpm)
    mapper = em.label_mapper()
    # Two mutations (one per site).
    assert len(mapper) == 2
    for value in mapper.values():
        assert isinstance(value, str) and len(value) >= 3


def test_epistasis_map_labels_with_gpm() -> None:
    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    sites = encoding_to_sites(order=2, encoding_table=gpm.encoding_table)
    em = EpistasisMap(sites=sites, gpm=gpm)
    labels = em.labels
    assert labels[0] == "w.t."
    assert all(isinstance(x, list) for x in labels[1:])


# assert_epistasis decorator.


def test_assert_epistasis_raises_when_missing() -> None:
    class Foo:
        @assert_epistasis
        def do(self) -> str:
            return "done"

    with pytest.raises(AttributeError):
        Foo().do()


def test_assert_epistasis_passes_when_present() -> None:
    class Foo:
        def __init__(self) -> None:
            self.epistasis = EpistasisMap(sites=[(0,)])

        @assert_epistasis
        def do(self) -> str:
            return "done"

    assert Foo().do() == "done"
