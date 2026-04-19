//! epistasis-core: Rust hot-path kernels for epistasis-v2.
//!
//! This crate is exposed to Python as `epistasis._core` via PyO3.
//! Phase 0 scaffold: no kernels implemented yet. Phase 2 adds
//! `build_model_matrix`, `encode_vectors`, and the Walsh-Hadamard
//! transform fast path.

use pyo3::prelude::*;

/// Returns the crate version. Smoke test for the Python-Rust bridge.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
