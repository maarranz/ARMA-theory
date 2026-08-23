# ARMA Theory

**ARMA Theory** is a small educational Python library for exploring the population properties and dynamic behaviour of autoregressive moving-average models.

The project is designed for teaching time series analysis and econometrics. It starts from a set of AR and MA coefficients and makes the implied mathematics visible: polynomial roots, causality, invertibility, theoretical correlations, innovation responses, and long-run effects. It does **not** estimate models, forecast observed data, or replace a full time-series package.

## Model convention

Throughout the project, an ARMA\((p,q)\) process is written as

$$
y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p}
      + \varepsilon_t
      + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}.
$$

Equivalently,

$$
\phi(L)y_t = \theta(L)\varepsilon_t,
$$

with

$$
\phi(L)=1-\phi_1L-\cdots-\phi_pL^p,
\qquad
\theta(L)=1+\theta_1L+\cdots+\theta_qL^q.
$$

The signs of the Python inputs follow this convention directly. For example,

```python
ar = [0.6, -0.2]
ma = [0.5]
```

represents

$$
y_t=0.6y_{t-1}-0.2y_{t-2}+\varepsilon_t+0.5\varepsilon_{t-1}.
$$

Use an empty list for an absent component: `ar=[]` for a pure MA model and `ma=[]` for a pure AR model.

## What the library provides

### Polynomial diagnostics

`arma_diagnostics()` constructs the AR and MA lag polynomials and reports:

- model order and validated coefficients;
- AR and MA roots;
- inverse roots;
- causality/stationarity of the AR component;
- invertibility of the MA component;
- approximate common-root matches; and
- whether the supplied representation is minimal.

The strict conditions used by the library are:

- the ARMA model is causal (and covariance-stationary) when every root of \(\phi(z)=0\) lies outside the unit circle;
- the MA representation is invertible when every root of \(\theta(z)=0\) lies outside the unit circle; and
- an ARMA representation is nonminimal when the AR and MA polynomials share a root, because a common factor can be cancelled.

Roots sufficiently close to the unit circle, or to one another, are classified using the configurable `root_tolerance`.

```python
import arma_theory as arma

diagnostics = arma.arma_diagnostics(
    ar=[0.6, -0.2],
    ma=[0.5],
)

print(diagnostics["causal"])
print(diagnostics["invertible"])
print(diagnostics["minimal_representation"])
print(diagnostics["inverse_ar_roots"])
print(diagnostics["inverse_ma_roots"])
```

### Theoretical ACF and PACF

For causal models, the library computes:

- theoretical autocovariances from a truncated infinite-MA representation;
- the theoretical autocorrelation function (ACF); and
- the theoretical partial autocorrelation function (PACF), using the Durbin–Levinson recursion.

```python
acf = arma.arma_acf(
    ar=[0.8],
    ma=[],
    n_lags=10,
)

pacf = arma.arma_pacf(
    ar=[0.8],
    ma=[],
    n_lags=10,
)
```

The figures show population correlations, so sampling confidence bands are intentionally omitted.

### ψ-weights and dynamic response

The transfer function

$$
\psi(L)=\frac{\theta(L)}{\phi(L)}
       =\sum_{j=0}^{\infty}\psi_jL^j
$$

describes how a one-unit innovation propagates through the process. The library calculates the formal recursion

$$
\psi_0=1,
\qquad
\psi_j=\theta_j+\sum_{i=1}^{\min(p,j)}\phi_i\psi_{j-i},
$$

where \(\theta_j=0\) beyond the MA order.

For a causal model, these weights form its convergent infinite-MA representation. For a unit-root or explosive model, the same finite recursion remains useful for displaying non-decaying or explosive dynamics, but it must not be interpreted as a convergent infinite-MA representation.

```python
psi = arma.arma_psi_weights(
    ar=[0.8],
    ma=[0.5],
    n_terms=11,
)

results = arma.arma_dynamics(
    ar=[0.8],
    ma=[0.5],
    horizon=20,
)

print(results["psi"])
print(results["cumulative_psi"])
print(results["long_run_response"])
```

`arma_dynamics()` combines the ψ-weights, their cumulative sum, model diagnostics, and long-run information in one result.

### Cumulative and long-run response

The cumulative response through horizon \(h\) is

$$
\sum_{j=0}^{h}\psi_j.
$$

For a causal model, its finite long-run limit is

$$
\sum_{j=0}^{\infty}\psi_j
=\psi(1)
=\frac{\theta(1)}{\phi(1)}.
$$

The library reports this analytical long-run response only when the model is causal. Unit-root and explosive cases can still be inspected over a finite horizon, but no finite long-run response is asserted.

For example, the ARMA\((1,1)\) model with `ar=[0.8]` and `ma=[0.5]` has

$$
\frac{1+0.5}{1-0.8}=7.5.
$$

## Visualizations

The library provides both static Matplotlib figures and interactive Plotly figures.

### Matplotlib

```python
arma.plot_arma_correlations(
    ar=[0.8],
    ma=[0.5],
    n_lags=20,
)

arma.plot_arma_dynamic_response(
    ar=[0.8],
    ma=[0.5],
    horizon=20,
)

arma.plot_arma_cumulative_response(
    ar=[0.8],
    ma=[0.5],
    horizon=20,
)
```

Separate Matplotlib ACF and PACF functions are also available as `plot_arma_acf()` and `plot_arma_pacf()`.

### Plotly

```python
arma.plotly_arma_correlations(
    ar=[0.8],
    ma=[0.5],
    n_lags=20,
)

arma.plotly_arma_dynamic_response(
    ar=[0.8],
    ma=[0.5],
    horizon=20,
)

arma.plotly_arma_cumulative_response(
    ar=[0.8],
    ma=[0.5],
    horizon=20,
)
```

Interactive figures add hover information while preserving the same theoretical quantities as the static versions. Separate Plotly ACF and PACF functions are available as `plotly_arma_acf()` and `plotly_arma_pacf()`.

### Inverse-root diagrams

Inverse-root diagrams place \(1/z\), rather than the polynomial root \(z\), in the complex plane. This makes the usual conditions visually immediate:

- causal AR inverse roots lie strictly inside the unit circle;
- invertible MA inverse roots lie strictly inside the unit circle; and
- coincident AR and MA inverse roots reveal a common factor and a nonminimal representation.

```python
arma.plot_arma_inverse_roots(
    ar=[1.2, -0.8],
    ma=[],
)

arma.plotly_arma_inverse_roots(
    ar=[0.5],
    ma=[-0.5],
)
```

The first example illustrates a complex-conjugate pair. In the second, the AR and MA polynomials share a root.

## Installation with Mamba

Clone the repository and enter its root directory:

```bash
git clone https://github.com/maarranz/ARMA-theory.git
cd ARMA-theory
```

Create and activate the environment defined in `environment.yml`:

```bash
mamba env create -f environment.yml
mamba activate arma_theory
```

Launch JupyterLab:

```bash
jupyter lab
```

Open `ARMA_Theory_Demo.ipynb`, or use the module directly from the repository root:

```python
import arma_theory as arma
```

The project is not yet packaged for installation with `pip`, so run notebooks and Python sessions from the repository root for now.

## A compact worked example

```python
import arma_theory as arma

ar = [0.8]
ma = [0.5]

# Polynomial properties
diagnostics = arma.arma_diagnostics(ar=ar, ma=ma)

# Population correlations
acf = arma.arma_acf(ar=ar, ma=ma, n_lags=20)
pacf = arma.arma_pacf(ar=ar, ma=ma, n_lags=20)

# Innovation dynamics through horizon 20
dynamics = arma.arma_dynamics(ar=ar, ma=ma, horizon=20)

print(f"Causal: {diagnostics['causal']}")
print(f"Invertible: {diagnostics['invertible']}")
print(f"Long-run response: {dynamics['long_run_response']:.2f}")

# Static and interactive views
arma.plot_arma_correlations(ar=ar, ma=ma, n_lags=20)
arma.plotly_arma_cumulative_response(ar=ar, ma=ma, horizon=20)
```

## Repository structure

The current structure is intentionally unchanged while a common packaging and documentation pattern is considered for this and related teaching projects.

```text
ARMA-theory/
├── arma_theory.py
├── ARMA_Theory_Demo.ipynb
├── environment.yml
├── README.md
├── LICENSE
└── tests/
```

## Project status

The numerical and visualization core is feature-complete as of **v0.5.5**. The current phase is documentation and teaching preparation rather than expansion of the mathematical API.

The demonstration notebook has been run successfully from a restarted kernel across representative cases, including AR, MA, and ARMA models; unit-root and explosive processes; complex roots; and common-root representations.

This is still a pre-1.0 educational project. Interfaces may be refined while the teaching materials and packaging are prepared.

## Roadmap

Near-term work is deliberately focused on presentation and maintainability:

- write a concept-led teaching notebook from scratch;
- retain the current demonstration notebook as a development and regression reference;
- review and harmonize docstrings and examples;
- add a minimal `pyproject.toml` and editable-install workflow;
- decide on a shared repository layout for `ARMA-theory` and related teaching libraries; and
- incorporate small refinements discovered through classroom use.

No major new core features are currently planned.

## Intended audience

ARMA Theory is intended for:

- undergraduate and postgraduate students in time series analysis or econometrics;
- instructors preparing lectures, demonstrations, and exercises;
- researchers who want transparent population-level ARMA calculations; and
- anyone interested in seeing how innovations propagate through a scalar dynamic model.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
