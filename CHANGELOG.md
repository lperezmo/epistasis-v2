# CHANGELOG


## v1.1.1 (2026-05-03)

### Bug Fixes

- **benchmarks**: Bind loop variables in lambdas and wrap long docstring line
  ([`d0875cf`](https://github.com/lperezmo/epistasis-v2/commit/d0875cfa9702810b8d8edec185386fabc839cef6))

### Chores

- Add badges, v1 vs v2 benchmark tables, and benchmarks/vs_v1.py
  ([`928ac55`](https://github.com/lperezmo/epistasis-v2/commit/928ac558624a039841abe5b1b11f125ee06f0ef6))

- **examples**: Add Streamlit showcase app
  ([`1e537c5`](https://github.com/lperezmo/epistasis-v2/commit/1e537c5273f557969a5da80e05a4bec8439ac118))

Multi-page demo under examples/, patterned after the streamlit-aggrid-v2 showcase. Seven pages:

- Overview: 3D hypercube of sequence space with phenotype-colored vertices, switchable between L=3
  and L=4 (tesseract projection). - Design matrix: interactive L / order / encoding, Plotly heatmaps
  of X and X^T X to show Hadamard orthogonality. - Linear fit: simulate known coefficients, fit,
  scatter predicted vs observed, coefficient bars with analytic OLS stderrs. - FWHT fast path: live
  benchmark of dense lstsq against fwht_ols_coefficients, asymptotic-cost curves, L slider capped at
  12. - Simulate: synthetic GPM from simulate_random_linear_gpm, phenotype histogram, genotype
  preview. - Cross validation: per-fold R^2 bars from epistasis.validate.k_fold. - About: install,
  links, credits.

Visual notes:

- Navigation via st.navigation in sidebar groups with Material icons. - Gradient h2 header via
  st.html instead of st.title. - Tailwind teal + indigo palette in a shared plots.py helper; rose
  for negative-signed heatmaps; consistent modebar config across figures. - st.markdown + st.caption
  for headings per project convention. - width="stretch" replaces use_container_width (deprecated
  after 2025-12-31).

Dependencies live only in examples/requirements.txt so they do not contaminate the core dev or CI
  loops. examples/README.md carries the Open-in-Streamlit badge and run instructions; the top-level
  README links to the hosted app at epistasis-v2.streamlit.app.

The lock-file touches are the semantic-release v1.1.0 version bump catching up locally.

- **streamlit**: Dark/light theme support, interaction site grid, examples dep group
  ([`cff6dd1`](https://github.com/lperezmo/epistasis-v2/commit/cff6dd16fbb411f3174768f7cf3e3e8d4f20c157))

### Continuous Integration

- Bump actions off Node.js 20 before the Sept 2026 cutoff
  ([`e579661`](https://github.com/lperezmo/epistasis-v2/commit/e57966192d64f339b730fee3ffb82f8c1984dfc8))

GitHub is forcing Node.js 24 as the default starting June 2026 and removing Node.js 20 from runners
  in September 2026. The current release leaves deprecation annotations on every run. Mirroring the
  versions streamlit-aggrid already moved to:

- actions/checkout: v4 -> v6 - actions/setup-python: v5 -> v6 - actions/upload-artifact: v4 -> v7 -
  actions/download-artifact: v4 -> v7 - astral-sh/setup-uv: v3 -> v8 - codecov/codecov-action: v4 ->
  v6

PyO3/maturin-action stays on v1 (still the current major) and dtolnay/rust-toolchain stays pinned to
  @stable.

- Pin setup-uv to v8.1.0
  ([`ad5cd96`](https://github.com/lperezmo/epistasis-v2/commit/ad5cd964cfe491379a81455173dcafbc092a649a))

astral-sh/setup-uv does not publish a floating v8 tag yet (only v7.x has major-minor shortcuts).
  Pinning to the full v8.1.0 tag so the resolver actually finds it.


## v1.1.0 (2026-04-20)

### Documentation

- Refresh README status lines for landed Phase 2 and Phase 3 work
  ([`fe431a8`](https://github.com/lperezmo/epistasis-v2/commit/fe431a8c7a68a8a0fe79f6e761e75c0bd753f593))

The FWHT fast path is in EpistasisLinearRegression, so the "not yet landed" qualifier on the
  Walsh-Hadamard bullet was stale. The sparse Lasso path is still pending and now reads as a
  memory-scaling concern for L >= 20, not a speed item. Also corrects the bench directory note.

### Features

- **core**: Port design-matrix kernels and FWHT to Rust
  ([`5e4c420`](https://github.com/lperezmo/epistasis-v2/commit/5e4c420cda5d75f61180e1eaaa3f7e20dbc5899a))

Rust crate epistasis-core now exposes encode_vectors, build_model_matrix, and fwht via PyO3. The
  NumPy backend becomes a reference-only oracle in _reference.py, used by the parity test suite to
  verify the Rust kernels on every fit.

- encode_vectors: parallel int8 conversion with a serial threshold at 2^18 cells so small inputs
  skip rayon overhead. - build_model_matrix: ragged sites packed as flat int64 indices plus int64
  offsets; parallel over output columns with a serial threshold at 2^15 cells. - fwht: iterative
  in-place butterfly, zero extra deps.

Measured on the Windows workstation: build_model_matrix is 3x to 6x faster than the NumPy reference
  across L from 8 to 16. Full-order OLS via FWHT wins qualitatively (see test_fwht.py; the model
  wiring lands separately).

Bumps the Rust workspace version to 1.0.0 and teaches python-semantic-release to keep Cargo.toml in
  sync with pyproject.toml on subsequent releases.

### Performance Improvements

- **linear**: Add FWHT fast path to EpistasisLinearRegression
  ([`e03791c`](https://github.com/lperezmo/epistasis-v2/commit/e03791c47cd95f5a4674a55a524cf85cad277a8c))

EpistasisLinearRegression.fit now detects full-order biallelic libraries with global (Hadamard)
  encoding and solves OLS in O(n log n) via the Fast Walsh-Hadamard Transform, skipping dense matrix
  construction and the sklearn solver. The check is bitmask-based: genotype bitmasks must cover [0,
  2^L) bijectively and site bitmasks must do the same. Any mismatch (partial library, truncated
  order, local encoding, multi-allelic positions, duplicate bits inside a site) returns None from
  fwht_ols_coefficients and the sklearn path takes over unchanged.

The new epistasis.fast module exposes fwht_ols_coefficients as a reusable utility. After a
  successful FWHT fit the code syncs the sklearn estimator's fitted attributes (coef_, intercept_,
  n_features_in_) so predict and score continue to work through the existing composition boundary.
  Standard errors are NaN because the system is exactly determined.

Benchmarked on the Windows workstation: full-order fit at L=10 drops from 292 ms (np.linalg.lstsq)
  to 0.78 ms, and L=12 drops from 15.4 s to 3.4 ms. The speedup is qualitative (log n vs n^2 in the
  solve) and scales further with L.


## v1.0.0 (2026-04-20)

### Build System

- Harden release config and add .gitattributes
  ([`3b61467`](https://github.com/lperezmo/epistasis-v2/commit/3b6146751074b98f6d8ff399f3a4f9bc22bebb81))

Applies lessons from the gpmap-v2 first release:

- Add allow_zero_version = false so the first semantic-release run jumps from 0.0.0 straight to
  1.0.0 instead of climbing through 0.0.1, 0.0.2, etc. This ignores the 0.1.0 placeholder in
  project.version; that's fine, semantic-release rewrites it anyway. - Add an explanatory comment
  above build_command = "" so nobody re-adds a build step here. The python-semantic-release job runs
  in a Docker image without Rust; wheels are built downstream in PyO3/maturin-action jobs where the
  toolchain is available. - Add .gitattributes with "* text=auto eol=lf" to normalize line endings.
  Silences the CRLF warnings Windows developers see on every commit.

- Switch gpmap-v2 dep from editable local to PyPI 1.0.0
  ([`b41b474`](https://github.com/lperezmo/epistasis-v2/commit/b41b4741dd3dd86afaac917811f8d9ce33ec9c9d))

gpmap-v2 is live on PyPI as of today. Dropping the [tool.uv.sources] entry that pointed at ../gpmap,
  pinning gpmap-v2>=1.0.0 in project.dependencies. Local checkouts and CI no longer require a
  sibling gpmap-v2 working tree to install.

All 172 tests still pass against PyPI gpmap-v2 1.0.0. The bridge contract tests in
  tests/test_gpmap_bridge.py continue to verify the public surface we consume.

### Documentation

- Enrich README with current progress and add CONTRIBUTING guide
  ([`2689925`](https://github.com/lperezmo/epistasis-v2/commit/2689925a41e2e7ac53bc2a86493e446c3d9ef37e))

README now lists the ported modules, the pending ones, the repo layout, and the dev loop commands.
  Points at CONTRIBUTING for anything more detailed.

CONTRIBUTING covers prerequisites (Python, Rust, uv, local gpmap-v2), the day-to-day loop,
  Conventional Commits conventions with a scope table keyed to the version-bump behavior we get from
  python-semantic-release, the PR checklist, and the automated release flow. Also documents the
  _legacy/ convention and how to coordinate schema changes with gpmap-v2.

- Relax CONTRIBUTING to match the vibe
  ([`30d02a7`](https://github.com/lperezmo/epistasis-v2/commit/30d02a7c612f3623b58dded5329a007fda7b6187))

Drops the rigid PR checklist and role-sounding language. Keeps what's actually load-bearing:
  Conventional Commits format, the dev setup commands, CI does the checking, no coauthor lines, no
  em dashes, no emojis. PRs are welcome if they make sense.

- Update install notes for the PyPI-pinned gpmap-v2
  ([`60a2897`](https://github.com/lperezmo/epistasis-v2/commit/60a2897796a0423ef7a7cb8d7aa7c9243e797262))

README: drop the sibling-checkout sentence, say gpmap-v2 comes from PyPI.

CONTRIBUTING: same, plus a short recipe for opting back into a local editable dep if someone wants
  to co-develop against a checkout of gpmap-v2.

### Features

- Port simulate, stats, validate, and Bayesian sampling
  ([`662b613`](https://github.com/lperezmo/epistasis-v2/commit/662b613102296b80e87a6d6af148ff6acc67eceb))

Finishes Phase 1 of the port. Suite now at 225 passing, mypy strict clean, ruff clean.

epistasis.simulate: - simulate_linear_gpm(wildtype, mutations, order, coefficients, ...) builds a
  GenotypePhenotypeMap whose phenotypes are X @ coefs. - simulate_random_linear_gpm samples
  coefficients uniformly from a user-supplied range, with optional rng for reproducibility. -
  Functional API: no BaseSimulation subclass hierarchy, no mutable .build() hooks. Call the function
  again to resimulate. - Ternary and higher-alphabet positions supported. power variant deferred
  pending models/nonlinear/power.py port.

epistasis.stats: - pearson, r_squared, rmsd, ss_residuals, aic, split_gpm. - Dropped v1 cruft:
  incremental_mean/var/std (unused), gmean (used only by deferred power transform), chi_squared
  (divide-by-pred explodes), explained_variance (redundant with r_squared),
  false_positive_rate/false_negative_rate (brittle and unused). - split_gpm builds fresh GPM
  instances from subsets via the constructor; no from_dataframe dance.

epistasis.validate: - k_fold(gpm, model, k, rng) returns R^2 scores. - holdout(gpm, model, fraction,
  repeat, rng) returns train and test score lists. Both take an optional np.random.Generator for
  reproducibility.

epistasis.sampling.bayesian: - BayesianSampler wraps a fitted model (SamplerModel Protocol: has
  thetas and lnlikelihood(thetas=)). Modernized for emcee 3: uses State objects, run_mcmc with
  positional initial state, reset() to drop burn-in samples. - initial_walkers seeds a Gaussian
  cloud around ML thetas with per-coefficient scale that falls back to relative_width when a
  coefficient is zero. - Subsequent sample() calls continue from the last state unless a fresh rng
  is supplied.

Tests: 53 new cases across the four modules. Covers simulation of predictable and random phenotypes
  with handbuilt-design-matrix verification, stats formulas and identities, GPM splitting by
  fraction or explicit train index with rng-reproducibility checks, k-fold and holdout contract
  tests on a simulated linear GPM, and sampler init guards, walker shapes, chain finiteness,
  continuation, and custom prior invocation.

README progress section fully updated.

- Scaffold epistasis-v2 with uv, maturin, and PyO3
  ([`f3a2f6c`](https://github.com/lperezmo/epistasis-v2/commit/f3a2f6ce7d9fd8f161e2af174eb40d873ee72897))

Clean-break rewrite of harmslab/epistasis. Phase 0 scaffold.

- pyproject.toml with maturin build backend - Rust workspace + epistasis-core crate exposed as
  epistasis._core - Python 3.10 through 3.13, abi3 wheels - GitHub Actions CI (lint, test, multi-OS,
  multi-Python) - GitHub Actions release workflow (python-semantic-release + maturin wheels + PyPI
  OIDC) - ruff, mypy, pytest, hypothesis dev stack - Smoke test validates Python-Rust bridge and
  version agreement

Legacy v1 source archived under _legacy/ (gitignored) for reference during porting. Full plan in
  scratchpad.md (gitignored).

- Wire gpmap-v2 as editable dep and add API contract tests
  ([`5eb1022`](https://github.com/lperezmo/epistasis-v2/commit/5eb10223bed7bfe111f6467a65dfb62b48ea7a58))

Uses uv sources to resolve gpmap-v2 to the sibling gpmap/ working tree via maturin editable install.
  Adds a contract test file that fails loudly if any gpmap-v2 public symbol we depend on disappears
  or changes shape: container API, encoding_table schema (site_index, site_label, wildtype_letter,
  mutation_letter, mutation_index), binary_packed dtype, subclassability, and the legacy
  genotype_index DeprecationWarning.

All 12 tests pass locally. Once gpmap-v2 publishes to PyPI we swap the uv.sources entry for a
  version pin.

- **mapping**: Port mapping module against gpmap-v2 schema
  ([`bf1af30`](https://github.com/lperezmo/epistasis-v2/commit/bf1af303626bbc77abfa6638277c1991043c9528))

Clean-break rewrite of epistasis/mapping.py. Reads site_index (not the deprecated genotype_index)
  from the encoding_table. Uses gpmap-v2 column names throughout.

Changes from v1: - Full type hints; mypy strict passes. - EpistasisMapReference collapsed into
  EpistasisMap.get_orders, which returns a fresh EpistasisMap instead of a view-like proxy. -
  Removed the no-op self.data.loc leftover in EpistasisMap._from_sites. - Setters validate length
  and coerce to float64; bad shapes raise ValueError instead of silently writing a DataFrame
  attribute. - Exceptions are TypeError/ValueError/AttributeError instead of bare Exception. - Added
  from_dataframe classmethod to match the gpmap-v2 convention. - __slots__ on EpistasisMap.

Tests: 33 new cases covering site_to_key/key_to_site roundtrip, genotype_coeffs orders,
  encoding_to_sites on biallelic and ternary tables and against a real GenotypePhenotypeMap,
  EpistasisMap construction/setters/validation/get_orders/label_mapper, to_csv roundtrip, and the
  assert_epistasis decorator.

Full suite now at 45 passing. Added pandas-stubs and a mypy override that silences missing-stub
  errors for gpmap until gpmap-v2 ships a py.typed marker.

- **matrix**: Port design-matrix construction with NumPy backend
  ([`8703bb3`](https://github.com/lperezmo/epistasis-v2/commit/8703bb3187f9840e5c1b2fe34f3d0f155c1d8233))

Clean-break rewrite of epistasis/matrix.py against gpmap-v2's binary_packed layout (uint8 2D, no
  string parse).

Public surface: - encode_vectors(binary_packed, model_type): returns an (n, L+1) int8 array with a
  leading intercept column. Global uses the Hadamard encoding (0 to +1, 1 to -1). Local keeps 0/1. -
  build_model_matrix(encoded, sites): dense design matrix. Each column is an elementwise product
  over the site's selected encoded columns. int8 output. - get_model_matrix(binary_packed, sites,
  model_type): convenience. - model_matrix_as_dataframe(matrix, sites, index): pandas wrapper.

This module is the correctness reference; a Rust kernel in epistasis._core replaces
  build_model_matrix and encode_vectors in Phase 2, keyed on the same signatures.

Input validation is strict: binary_packed must be uint8 2D with values in {0, 1}; sites must be
  non-empty tuples whose indices fit the encoded width; model_type must be 'global' or 'local'. Bad
  shapes or values raise ValueError.

Tests: 28 new cases covering encoding for both model types, intercept column invariants,
  product-of-columns correctness at orders 1/2/3, error paths on bad inputs, integration with a real
  GenotypePhenotypeMap, DataFrame wrapper, and a parametrized Hadamard orthogonality check (X^T @ X
  = 2^L * I) for L in {2, 3, 4, 5}.

Suite now at 73 passing.

- **models**: Port abstract and base model with composition over MRO injection
  ([`fd2449a`](https://github.com/lperezmo/epistasis-v2/commit/fd2449a714b053b5e2c9c279f5717363295c8eb5))

Replaces v1's @use_sklearn decorator (which meta-programmed sklearn classes into the MRO) with plain
  attribute composition. Concrete models will hold a sklearn/lmfit estimator as a field and forward
  explicitly. This unlocks modern sklearn (>=1.2) which broke the MRO trick when normalize= was
  removed.

New modules: - epistasis.exceptions: EpistasisError, XMatrixError, FittingError (renamed from v1's
  *Exception for PEP 8 consistency). - epistasis.utils: genotypes_to_X helper, now reading
  binary_packed from gpmap-v2 (not the string binary). - epistasis.models.base:
  AbstractEpistasisModel (3 abstract methods: fit, predict, hypothesis) and EpistasisBaseModel
  (concrete foundation).

EpistasisBaseModel carries: - add_gpm(gpm) wires the model to a GenotypePhenotypeMap, builds site
  columns via encoding_to_sites, and creates the EpistasisMap. - gpm / epistasis / Xcolumns
  properties raise XMatrixError with a clear message if accessed before add_gpm(). -
  _resolve_X/y/yerr handle None (use attached GPM), 2D arrays (assumed design matrices), and
  iterables of genotype strings. Rejects everything else explicitly. - lnlike_of_data default is
  Gaussian (mean = hypothesis, sd = yerr). Subclasses with non-Gaussian noise override. Suppresses
  noisy RuntimeWarnings inside np.errstate; lnlikelihood collapses non-finite totals to -inf. -
  predict_to_df/csv/excel require genotype strings (rejects raw design matrices because they lack
  labels).

Dropped from v1: @use_sklearn, arghandler (inspect-based arg injector), AbstractModel.__new__
  docstring copy hack, Bunch, DocstringMeta (both had Python 2 bugs), and the broken
  single-string-genotype dispatch.

Tests: 26 new cases using an IdentityLinearModel subclass to exercise

every base-class behavior: attach/detach, property guards, resolver paths, Gaussian likelihood shape
  and -inf collapse, predict output helpers, and cache invalidation on re-add_gpm.

Suite now at 99 passing. pandas-stubs happy, mypy strict clean, ruff clean.

- **models**: Port EpistasisLinearRegression with analytic OLS stderr
  ([`ca0fb75`](https://github.com/lperezmo/epistasis-v2/commit/ca0fb750155aab435073c6389d07992e5aad0250))

First concrete model. Uses composition: holds a sklearn LinearRegression configured for
  fit_intercept=False (the intercept lives in the design matrix as site (0,)) and forwards
  fit/predict/score explicitly. No more @use_sklearn MRO injection, no more arghandler.

The big new feature is analytic coefficient standard errors (issue #56). On fit(), the model
  computes sigma_hat^2 * (X'X)^-1 via pinv and stores sqrt(diag(cov)) into
  self.epistasis.stdeviations. When n <= p (underdetermined), stderr stays NaN. Full OLS uncertainty
  quantification for free.

Fit/predict/hypothesis/score all accept the full resolver surface: None (use attached GPM), 2D
  design matrix, or iterable of genotype strings. Raises FittingError with a clear message when
  called before fit().

Tests: 19 new cases. Covers construction, guards before fit, exact recovery at full order,
  epistasis.values/stdeviations population, coef_ alias, score==1 on exact fit, prediction on
  genotype subsets, hypothesis with custom thetas, likelihood computation after fit, stderr NaN when
  not overdetermined, stderr finite and matching the manual sigma_hat * sqrt(diag((X'X)^-1)) formula
  when overdetermined, and raw design matrix passthrough.

Also added sklearn, lmfit, emcee to the mypy missing-stub ignore list since none of them ship
  py.typed.

Suite now at 118 passing. mypy strict clean, ruff clean.

- **models**: Port EpistasisLogisticRegression for viability classification
  ([`08414cc`](https://github.com/lperezmo/epistasis-v2/commit/08414cc4e54d730af16b41cf7da464cd29d7e31e))

Ports the logistic classifier from v1 with composition. Drops the specialized v1 classes that were
  either stubs (QuadraticDiscriminant, whose methods all pass) or duplicative (SVC, BernoulliNB
  imports that were never wired up) or niche (GaussianProcess, GMM). They can come back later if
  someone actually needs them.

New modules:

- classifiers/_base.py: EpistasisClassifierBase, shared foundation. Holds an Additive (order-1
  EpistasisLinearRegression) and a subclass-supplied _sklearn classifier. Implements _projected_X
  which scales each column of the additive design matrix by its fitted additive coefficient,
  matching the v1 feature transform. Provides default
  fit/predict/predict_proba/predict_log_proba/score on top of that projection.

- classifiers/logistic.py: EpistasisLogisticRegression, the concrete classifier. Binarizes observed
  phenotypes at `threshold`, fits the projected features. lnlike_of_data overrides the base Gaussian
  default with a Bernoulli log-likelihood. Dropped the `penalty` parameter since sklearn 1.8
  deprecated it; users wanting L1 can assign a custom sklearn LogisticRegression to self._sklearn.

Fixes v1 confusion: v1's hypothesis took raw X (not projected) while predict took projected X,
  making them inconsistent. v2 hypothesis returns predict_proba(X)[:, 1] so predict and hypothesis
  agree.

Tests: 17 cases. Covers construction guards, fit round-trip, additive wiring, score >= 0.7 on
  structured viability data, predict_proba rows sum to 1, predict_log_proba matches
  log(predict_proba), hypothesis returns P(class=1), per-sample lnlike is finite and non-positive,
  lnlike under thetas=0 equals log(0.5) exactly, lnlikelihood sums correctly, _binarize_y honors the
  threshold, and higher thresholds move samples into class 0.

README progress section updated.

Suite now at 189 passing. mypy strict clean, ruff clean.

- **models**: Port Ridge, Lasso, and ElasticNet with shared composition base
  ([`f1ff223`](https://github.com/lperezmo/epistasis-v2/commit/f1ff223202f857370ab41effaf40ba6ea8e179f1))

Three regularized linear models, refactored onto a common RegularizedLinearBase in _regularized.py
  that carries the shared fit/predict/hypothesis/score/coef_/compression_ratio machinery. Subclasses
  only configure the right sklearn estimator in __init__. Much less duplication than v1.

Fixes v1 ElasticNet bug: __init__ silently overwrote l1_ratio=user_value with l1_ratio=1.0, making
  ElasticNet behave as pure Lasso regardless of what the user passed. v2 threads the parameter
  straight into sklearn.ElasticNet and validates the [0, 1] range.

EpistasisRidge: - L2-only penalty. alpha=0 degenerates to OLS (test proves it). - compression_ratio
  always 0 since L2 shrinks without zeroing (test confirms this).

EpistasisLasso: - L1 penalty, produces exact zeros. - compression_ratio monotone in alpha (test). -
  positive=True enforces non-negativity (test).

EpistasisElasticNet: - Mixed L1 + L2. l1_ratio=1.0 matches Lasso exactly (test); intermediate ratios
  diverge from both Lasso and Ridge (test). - ValueError on l1_ratio outside [0, 1]. - Regression
  test asserts self._sklearn.l1_ratio == user_ratio across {0.0, 0.25, 0.5, 0.75, 1.0}.

Tests: 34 new cases, parametrized across all three models where the behavior is shared, plus
  model-specific invariants.

README progress section updated.

Suite now at 152 passing. mypy strict clean, ruff clean.

- **models**: Port two-stage nonlinear epistasis regression
  ([`be2f560`](https://github.com/lperezmo/epistasis-v2/commit/be2f5605cf5a067d24b43c802e47521d716ddbdc))

Ports the core nonlinear machinery from v1. Two-stage fit: an order-1 linear additive model
  approximates the average per-mutation effect, then a user-supplied nonlinear function f(x,
  *params) is fit so that f(Additive.predict(X)) approximates the observed phenotypes. Uses
  lmfit.minimize for Levenberg-Marquardt optimization.

New modules:

- epistasis.models.nonlinear.minimizer: Minimizer ABC and FunctionMinimizer. Cleaner than v1:
  introspects parameter names via inspect.signature (first parameter must be 'x'), takes explicit
  initial_guesses dict instead of **p0 kwargs, defaults unspecified parameters to 1.0 instead of
  None (which lmfit misbehaves on), raises FittingError on lmfit failure instead of the v1
  print-and- reraise dance.

- epistasis.models.nonlinear.ordinary: EpistasisNonlinearRegression. Composes an
  EpistasisLinearRegression (the additive stage) with a FunctionMinimizer (the nonlinear stage).
  Clean add_gpm that fans out to the additive submodel. thetas concatenates nonlinear values then
  linear coefficients. hypothesis accepts full-length thetas and splits them. transform linearizes
  observed phenotypes back onto the additive scale for downstream linear analysis. score returns
  Pearson R^2.

Deferred to a later commit: power.py (power-transform minimizer) and spline.py (spline minimizer).
  Both are specialized subclasses of the nonlinear core.

Tests: 20 cases. Covers FunctionMinimizer construction (name intro-spection, first-arg-is-x
  rejection, initial-guess handling, default of 1.0), fit recovery of known (A, K) on clean data,
  predict/trans-form formulas, end-to-end EpistasisNonlinearRegression phenotype reconstruction on a
  synthetic GPM built from a saturation curve, thetas shape and hypothesis round-trip, score in the
  expected range, predict on genotype subsets, and the additive-X slicing path that accepts a
  higher-order design matrix and uses only its first-order columns.

Suite now at 172 passing. mypy strict clean, ruff clean.
