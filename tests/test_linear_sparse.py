"""Tests for the sparse design-matrix path on Lasso / ElasticNet.

The sparse path keeps the design matrix out of dense float64 for `local`
encoding (and on demand for `global`). These tests verify equivalence with
the dense path, that `sparse="auto"` engages only for `local` encoding, and
that prediction works through the sparse pipeline.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from epistasis.matrix import (
    build_model_matrix,
    build_model_matrix_sparse,
    encode_vectors,
    get_model_matrix_sparse,
)
from epistasis.models.linear import EpistasisElasticNet, EpistasisLasso
from gpmap import GenotypePhenotypeMap
from scipy.sparse import csc_matrix, issparse


def _gpm_nsite(n: int, seed: int = 0) -> GenotypePhenotypeMap:
    genotypes = np.array(["".join(g) for g in product("AB", repeat=n)])
    rng = np.random.default_rng(seed)
    phenotypes = rng.normal(size=len(genotypes))
    return GenotypePhenotypeMap(
        wildtype="A" * n,
        genotypes=genotypes,
        phenotypes=phenotypes,
    )


@pytest.fixture
def gpm4() -> GenotypePhenotypeMap:
    return _gpm_nsite(4, seed=2)


# ----------------------------------------------------------------------
# Sparse matrix builder.


@pytest.mark.parametrize("model_type", ["global", "local"])
def test_sparse_builder_matches_dense(
    model_type: str,
    gpm4: GenotypePhenotypeMap,
) -> None:
    from epistasis.mapping import encoding_to_sites

    sites = encoding_to_sites(2, gpm4.encoding_table)
    encoded = encode_vectors(gpm4.binary_packed, model_type=model_type)  # type: ignore[arg-type]
    dense = build_model_matrix(encoded, sites)
    sparse = build_model_matrix_sparse(encoded, sites, model_type=model_type)  # type: ignore[arg-type]
    assert isinstance(sparse, csc_matrix)
    np.testing.assert_array_equal(sparse.toarray().astype(np.int8), dense)


def test_sparse_builder_local_has_no_negatives(gpm4: GenotypePhenotypeMap) -> None:
    """Local encoding columns are 0/1 only, never -1. The sparse builder
    must encode exactly that range.
    """
    from epistasis.mapping import encoding_to_sites

    sites = encoding_to_sites(3, gpm4.encoding_table)
    sparse = get_model_matrix_sparse(gpm4.binary_packed, sites, model_type="local")
    assert (sparse.data == 1.0).all()


def test_get_model_matrix_sparse_shape(gpm4: GenotypePhenotypeMap) -> None:
    from epistasis.mapping import encoding_to_sites

    sites = encoding_to_sites(2, gpm4.encoding_table)
    sp = get_model_matrix_sparse(gpm4.binary_packed, sites, model_type="local")
    assert sp.shape == (len(gpm4.genotypes), len(sites))


def test_sparse_builder_rejects_wrong_dtype(gpm4: GenotypePhenotypeMap) -> None:
    from epistasis.mapping import encoding_to_sites

    sites = encoding_to_sites(2, gpm4.encoding_table)
    encoded = encode_vectors(gpm4.binary_packed, model_type="local").astype(np.int16)
    with pytest.raises(ValueError, match="dtype int8"):
        build_model_matrix_sparse(encoded, sites, model_type="local")


# ----------------------------------------------------------------------
# Auto-engagement of the sparse path.


@pytest.mark.parametrize("cls", [EpistasisLasso, EpistasisElasticNet])
def test_sparse_auto_engages_for_local(
    cls: type,
    gpm4: GenotypePhenotypeMap,
) -> None:
    m = cls(order=2, model_type="local", alpha=0.05).add_gpm(gpm4)
    assert m._use_sparse() is True


@pytest.mark.parametrize("cls", [EpistasisLasso, EpistasisElasticNet])
def test_sparse_auto_stays_off_for_global(
    cls: type,
    gpm4: GenotypePhenotypeMap,
) -> None:
    """Global Hadamard encoding has every cell nonzero so sparse is wasteful."""
    m = cls(order=2, model_type="global", alpha=0.05).add_gpm(gpm4)
    assert m._use_sparse() is False


@pytest.mark.parametrize("cls", [EpistasisLasso, EpistasisElasticNet])
def test_sparse_explicit_overrides_auto(
    cls: type,
    gpm4: GenotypePhenotypeMap,
) -> None:
    m_on = cls(order=2, model_type="global", alpha=0.05, sparse=True).add_gpm(gpm4)
    m_off = cls(order=2, model_type="local", alpha=0.05, sparse=False).add_gpm(gpm4)
    assert m_on._use_sparse() is True
    assert m_off._use_sparse() is False


# ----------------------------------------------------------------------
# Numerical equivalence: sparse fit matches dense fit.


@pytest.mark.parametrize("model_type", ["global", "local"])
def test_lasso_sparse_matches_dense_coefs(
    model_type: str,
    gpm4: GenotypePhenotypeMap,
) -> None:
    sparse = (
        EpistasisLasso(order=2, model_type=model_type, alpha=0.05, sparse=True).add_gpm(gpm4).fit()
    )
    dense = (
        EpistasisLasso(order=2, model_type=model_type, alpha=0.05, sparse=False).add_gpm(gpm4).fit()
    )
    np.testing.assert_allclose(sparse.thetas, dense.thetas, atol=1e-8)


@pytest.mark.parametrize("model_type", ["global", "local"])
def test_elastic_net_sparse_matches_dense_coefs(
    model_type: str,
    gpm4: GenotypePhenotypeMap,
) -> None:
    sparse = (
        EpistasisElasticNet(order=2, model_type=model_type, alpha=0.05, l1_ratio=0.5, sparse=True)
        .add_gpm(gpm4)
        .fit()
    )
    dense = (
        EpistasisElasticNet(order=2, model_type=model_type, alpha=0.05, l1_ratio=0.5, sparse=False)
        .add_gpm(gpm4)
        .fit()
    )
    np.testing.assert_allclose(sparse.thetas, dense.thetas, atol=1e-8)


# ----------------------------------------------------------------------
# Sparse path keeps prediction working.


def test_sparse_predict_matches_dense(gpm4: GenotypePhenotypeMap) -> None:
    s = EpistasisLasso(order=2, model_type="local", alpha=0.05).add_gpm(gpm4).fit()
    d = EpistasisLasso(order=2, model_type="local", alpha=0.05, sparse=False).add_gpm(gpm4).fit()
    np.testing.assert_allclose(s.predict(), d.predict(), atol=1e-10)


def test_sparse_score_matches_dense(gpm4: GenotypePhenotypeMap) -> None:
    s = EpistasisLasso(order=2, model_type="local", alpha=0.05).add_gpm(gpm4).fit()
    d = EpistasisLasso(order=2, model_type="local", alpha=0.05, sparse=False).add_gpm(gpm4).fit()
    assert s.score() == pytest.approx(d.score(), abs=1e-10)


def test_sparse_hypothesis_with_thetas(gpm4: GenotypePhenotypeMap) -> None:
    """Custom-theta hypothesis() works through the sparse design matrix."""
    s = EpistasisLasso(order=2, model_type="local", alpha=0.05).add_gpm(gpm4).fit()
    thetas = np.zeros_like(s.thetas)
    thetas[0] = 1.5
    out = s.hypothesis(thetas=thetas)
    assert out.shape == (len(gpm4.genotypes),)
    # Intercept column is all ones; constant offset of 1.5 expected.
    np.testing.assert_allclose(out, np.full(len(gpm4.genotypes), 1.5), atol=1e-10)


def test_sparse_predict_on_genotype_strings(gpm4: GenotypePhenotypeMap) -> None:
    s = EpistasisLasso(order=2, model_type="local", alpha=0.05).add_gpm(gpm4).fit()
    g = list(gpm4.genotypes[:3])
    out = s.predict(X=g)
    assert out.shape == (3,)


# ----------------------------------------------------------------------
# Accept user-supplied sparse X.


def test_sparse_accepts_user_csc(gpm4: GenotypePhenotypeMap) -> None:
    from epistasis.mapping import encoding_to_sites

    sites = encoding_to_sites(2, gpm4.encoding_table)
    Xsp = get_model_matrix_sparse(gpm4.binary_packed, sites, model_type="local")
    m = EpistasisLasso(order=2, model_type="local", alpha=0.05, sparse=True).add_gpm(gpm4)
    m.fit(X=Xsp, y=gpm4.phenotypes)
    # Same result as auto-built sparse path.
    auto = EpistasisLasso(order=2, model_type="local", alpha=0.05).add_gpm(gpm4).fit()
    np.testing.assert_allclose(m.thetas, auto.thetas, atol=1e-10)


# ----------------------------------------------------------------------
# Memory characteristic check at moderate L.


def test_sparse_density_is_low_at_high_order_local() -> None:
    """At order 3 with L=8 under local encoding, the design matrix is sparse
    enough that storing it as CSC saves real memory. This is the property
    the whole feature exists for.
    """
    from epistasis.mapping import encoding_to_sites

    gpm = _gpm_nsite(8, seed=3)
    sites = encoding_to_sites(3, gpm.encoding_table)
    sp = get_model_matrix_sparse(gpm.binary_packed, sites, model_type="local")
    density = sp.nnz / (sp.shape[0] * sp.shape[1])
    assert density < 0.30, f"local encoding at order=3, L=8 should be <30% dense; got {density:.2%}"


def test_sparse_design_matrix_is_sparse_format(gpm4: GenotypePhenotypeMap) -> None:
    m = EpistasisLasso(order=2, model_type="local", alpha=0.05).add_gpm(gpm4)
    X = m._resolve_X_for_solver(None)
    assert issparse(X)
