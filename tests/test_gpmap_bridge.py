"""Validates that gpmap-v2 is importable and exposes the surface epistasis-v2 needs.

This file is a contract check. If any symbol disappears or changes signature,
tests here fail and we renegotiate with the gpmap-v2 repo.
"""

import numpy as np
import pandas as pd
import pytest


def test_gpmap_imports() -> None:
    import gpmap

    assert hasattr(gpmap, "__version__")


def test_core_container_symbols() -> None:
    from gpmap import GenotypePhenotypeMap

    assert callable(getattr(GenotypePhenotypeMap, "from_dataframe", None))


def test_encoding_functions() -> None:
    from gpmap import genotypes_to_binary, genotypes_to_binary_packed

    assert callable(genotypes_to_binary)
    assert callable(genotypes_to_binary_packed)


def test_error_transforms() -> None:
    from gpmap.errors import lower_transform, upper_transform

    assert callable(upper_transform)
    assert callable(lower_transform)


def test_rust_extension_loaded() -> None:
    from gpmap import _rust

    assert hasattr(_rust, "genotypes_to_binary_packed")


def test_minimal_gpm_construction() -> None:
    from gpmap import GenotypePhenotypeMap

    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    assert gpm.wildtype == "AA"
    assert len(gpm.genotypes) == 4
    assert gpm.phenotypes.dtype == np.float64


def test_encoding_table_schema() -> None:
    """epistasis-v2 reads these columns from encoding_table."""
    from gpmap import GenotypePhenotypeMap

    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    et = gpm.encoding_table
    assert isinstance(et, pd.DataFrame)
    required = {
        "site_index",
        "site_label",
        "wildtype_letter",
        "mutation_letter",
        "mutation_index",
    }
    missing = required - set(et.columns)
    assert not missing, f"encoding_table missing columns: {missing}"


def test_binary_packed_is_uint8_2d() -> None:
    """Rust kernel in epistasis-v2 consumes binary_packed directly."""
    from gpmap import GenotypePhenotypeMap

    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    bp = gpm.binary_packed
    assert bp.dtype == np.uint8
    assert bp.ndim == 2
    assert bp.shape[0] == 4


def test_subclassable() -> None:
    """simulate/ ports will subclass GenotypePhenotypeMap."""
    from gpmap import GenotypePhenotypeMap

    class MySim(GenotypePhenotypeMap):
        pass

    sim = MySim(
        wildtype="A",
        genotypes=np.array(["A", "B"]),
        phenotypes=np.array([0.0, 1.0]),
    )
    assert isinstance(sim, GenotypePhenotypeMap)


def test_legacy_genotype_index_alias_warns() -> None:
    """During migration we may still read the v1 column name. It must warn."""
    from gpmap import GenotypePhenotypeMap

    gpm = GenotypePhenotypeMap(
        wildtype="AA",
        genotypes=np.array(["AA", "AB", "BA", "BB"]),
        phenotypes=np.array([0.0, 0.1, 0.2, 0.4]),
    )
    with pytest.warns(DeprecationWarning):
        _ = gpm.encoding_table["genotype_index"]
