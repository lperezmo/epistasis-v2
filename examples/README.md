# epistasis-v2 showcase

Multi-page Streamlit tour of the epistasis-v2 package.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://epistasis-v2.streamlit.app/)

## Run locally

```bash
pip install -r examples/requirements.txt
streamlit run examples/showcase.py
```

Or with `uv` and no permanent install:

```bash
uv run --with streamlit --with plotly streamlit run examples/showcase.py
```

## Pages

- **Overview**: sequence-space primer with a 3D hypercube of the library.
- **Design matrix**: pick L, order, and encoding; inspect X and X^T X.
- **Linear fit**: simulate, fit, verify against ground truth.
- **FWHT fast path**: benchmark dense lstsq against the Walsh-Hadamard solver.
- **Simulate**: build a synthetic GPM from configurable coefficients.
- **Cross validation**: per-fold R^2 on held-out phenotypes.
- **About**: install, links, credits.
