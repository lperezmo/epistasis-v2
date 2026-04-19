"""Tests for epistasis.models.base."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from epistasis.exceptions import XMatrixError
from epistasis.mapping import EpistasisMap
from epistasis.models.base import AbstractEpistasisModel, EpistasisBaseModel
from gpmap import GenotypePhenotypeMap


# A minimal concrete subclass for exercising the base class machinery.
class IdentityLinearModel(EpistasisBaseModel):
    """Fits y = X @ thetas via pseudoinverse. Minimal concrete model for tests."""

    def __init__(self, order: int = 1, model_type: str = "global") -> None:
        super().__init__(order=order, model_type=model_type)  # type: ignore[arg-type]
        self.thetas: np.ndarray | None = None

    def fit(self, X: Any = None, y: Any = None) -> IdentityLinearModel:
        X_mat = self._resolve_X(X).astype(np.float64)
        y_arr = self._resolve_y(y)
        self.thetas, *_ = np.linalg.lstsq(X_mat, y_arr, rcond=None)
        if self._epistasis is not None:
            self._epistasis.values = self.thetas
        return self

    def predict(self, X: Any = None) -> np.ndarray:
        if self.thetas is None:
            raise RuntimeError("Call fit() before predict().")
        X_mat = self._resolve_X(X).astype(np.float64)
        return X_mat @ self.thetas

    def hypothesis(
        self,
        X: Any = None,
        thetas: np.ndarray | None = None,
    ) -> np.ndarray:
        X_mat = self._resolve_X(X).astype(np.float64)
        th = thetas if thetas is not None else self.thetas
        if th is None:
            raise RuntimeError("thetas unavailable; fit or pass thetas=.")
        return X_mat @ th


# Fixture: 2-site biallelic GPM.
@pytest.fixture
def gpm_2site() -> GenotypePhenotypeMap:
    return GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.3, 0.5, 1.2]),
        stdeviations=np.array([0.1, 0.1, 0.1, 0.1]),
    )


def test_abstract_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        AbstractEpistasisModel()  # type: ignore[abstract]


def test_base_model_defaults() -> None:
    m = IdentityLinearModel(order=2)
    assert m.order == 2
    assert m.model_type == "global"


def test_gpm_accessor_errors_before_attach() -> None:
    m = IdentityLinearModel()
    with pytest.raises(XMatrixError, match="GenotypePhenotypeMap"):
        _ = m.gpm


def test_epistasis_accessor_errors_before_attach() -> None:
    m = IdentityLinearModel()
    with pytest.raises(XMatrixError, match="EpistasisMap"):
        _ = m.epistasis


def test_xcolumns_accessor_errors_before_attach() -> None:
    m = IdentityLinearModel()
    with pytest.raises(XMatrixError, match="Xcolumns"):
        _ = m.Xcolumns


def test_add_gpm_wires_everything(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    assert m.gpm is gpm_2site
    assert isinstance(m.epistasis, EpistasisMap)
    assert m.Xcolumns[0] == (0,)
    assert m.num_of_params == len(m.Xcolumns)


def test_resolve_X_none_uses_gpm(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    X1 = m._resolve_X(None)
    X2 = m._resolve_X(None)
    assert X1 is X2  # cached
    assert X1.shape == (4, len(m.Xcolumns))


def test_resolve_X_passthrough_2d(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    custom = np.ones((3, 4), dtype=np.int8)
    out = m._resolve_X(custom)
    assert out is custom


def test_resolve_X_from_genotype_list(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    X = m._resolve_X(["AA", "BB"])
    assert X.shape == (2, len(m.Xcolumns))


def test_resolve_X_rejects_non_strings(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    with pytest.raises(XMatrixError, match="genotype strings"):
        m._resolve_X([1, 2, 3])


def test_resolve_X_rejects_weird_ndim(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    with pytest.raises(XMatrixError, match="1D|2D"):
        m._resolve_X(np.zeros((2, 2, 2)))


def test_resolve_y_none_uses_gpm(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    y = m._resolve_y(None)
    assert y.dtype == np.float64
    np.testing.assert_allclose(y, [0.0, 0.3, 0.5, 1.2])


def test_resolve_y_coerces_to_float(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    y = m._resolve_y([1, 2, 3])
    assert y.dtype == np.float64


def test_resolve_yerr_none_uses_gpm(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    yerr = m._resolve_yerr(None)
    np.testing.assert_allclose(yerr, [0.1, 0.1, 0.1, 0.1])


# Fit / predict on the identity model.


def test_identity_fit_recovers_phenotypes_exactly(gpm_2site: GenotypePhenotypeMap) -> None:
    """Full-order OLS on 4 genotypes with 4 coefs is an exact fit."""
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    pred = m.predict()
    np.testing.assert_allclose(pred, gpm_2site.phenotypes, atol=1e-10)


def test_identity_fit_stores_coefficients(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    assert m.thetas is not None
    assert not np.any(np.isnan(m.epistasis.values))


def test_identity_predict_on_subset_of_genotypes(
    gpm_2site: GenotypePhenotypeMap,
) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    pred = m.predict(X=["AA", "BB"])
    np.testing.assert_allclose(pred, [0.0, 1.2], atol=1e-10)


def test_identity_hypothesis_with_custom_thetas(
    gpm_2site: GenotypePhenotypeMap,
) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    thetas = np.array([1.0, 0.0, 0.0, 0.0])  # constant model
    out = m.hypothesis(thetas=thetas)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0])


# Likelihood helpers.


def test_lnlike_of_data_gaussian_shape(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    ll = m.lnlike_of_data()
    assert ll.shape == (4,)


def test_lnlikelihood_finite_for_good_fit(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    ll = m.lnlikelihood()
    assert np.isfinite(ll)


def test_lnlikelihood_negative_inf_for_zero_yerr(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    yerr = np.zeros(4)
    ll = m.lnlikelihood(yerr=yerr)
    assert ll == float("-inf")


# Prediction output helpers.


def test_predict_to_df_default(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    df = m.predict_to_df()
    assert list(df.columns) == ["genotypes", "phenotypes"]
    assert len(df) == 4


def test_predict_to_df_with_custom_genotypes(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    df = m.predict_to_df(X=["AA", "BB"])
    assert list(df["genotypes"]) == ["AA", "BB"]


def test_predict_to_df_rejects_design_matrix(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    with pytest.raises(XMatrixError, match="lack labels"):
        m.predict_to_df(X=np.zeros((2, 4), dtype=np.int8))


def test_predict_to_csv_roundtrip(
    gpm_2site: GenotypePhenotypeMap,
    tmp_path: Any,
) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site).fit()
    path = tmp_path / "preds.csv"
    m.predict_to_csv(str(path))
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["genotypes", "phenotypes"]
    assert len(loaded) == 4


# add_gpm is idempotent and resets cache.


def test_add_gpm_resets_Xbuilt_cache(gpm_2site: GenotypePhenotypeMap) -> None:
    m = IdentityLinearModel(order=2).add_gpm(gpm_2site)
    _ = m._resolve_X(None)
    assert "default" in m._Xbuilt
    m.add_gpm(gpm_2site)
    assert "default" not in m._Xbuilt
