# ARMA Theory

**ARMA Theory** is a small educational Python library for exploring the population properties and dynamic behaviour of autoregressive moving-average (ARMA) models.

The library starts from user-specified AR and MA coefficients and computes the theoretical properties implied by those coefficients, including:

- causality and stationarity;
- invertibility;
- AR and MA polynomial roots and inverse roots;
- common roots and minimal representations;
- theoretical autocovariances, ACF and PACF;
- $\psi$-weights and the dynamic response to innovations;
- cumulative and long-run responses when they are defined.

The emphasis is on understanding how the coefficients of an ARMA model determine its theoretical behaviour. The library does **not** estimate ARMA models from data or provide forecasting methods. For those tasks, a general time-series package should be used.


## Prerequisites and Installation

ARMA Theory requires **Python 3.13 or later**. The repository includes an `environment.yml` file containing the Python dependencies needed to run the library and its examples.

The recommended setup uses **Miniforge, Conda, or Mamba** with the `conda-forge` channel. Mamba is used in the commands below, but the equivalent Conda commands can also be used.

Clone the repository and create the supplied environment:

```bash
git clone https://github.com/maarranz/ARMA-theory.git
cd ARMA-theory
mamba env create -f environment.yml
mamba activate arma_theory
```

The environment includes NumPy, SciPy, pandas, Matplotlib, Plotly, IPython/Jupyter, and the other packages required by the project.

ARMA Theory is not currently distributed as an installable Python package. Run Python or Jupyter from the repository directory and import the module directly:

```python
import arma_theory as arma
```

To verify that the module is available:

```python
arma.arma_diagnostics(ar=[0.8], ma=[])
```

Users who already have a suitable Python environment may install the required dependencies themselves instead of creating the supplied environment.

### JupyterLab Desktop

If you use JupyterLab Desktop, select the `arma_theory` environment in the application's Python environment manager. If the environment is not detected automatically, add it manually by selecting the Python executable from the `arma_theory` environment.

The supplied environment includes both JupyterLab and `ipykernel`, so no additional kernel installation should normally be necessary.


## Model convention

Throughout the project, an ARMA\((p,q)\) process is written as

$$
y_t = \varphi_1 y_{t-1} + \cdots + \varphi_p y_{t-p}
      + \varepsilon_t
      + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}.
$$

Equivalently,

$$
\varphi(L)y_t = \theta(L)\varepsilon_t,
$$

with

$$
\varphi(L)=1-\varphi_1L-\cdots-\varphi_pL^p,
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


## Polynomial Diagnostics

`arma_diagnostics()` examines the AR and MA polynomials of a specified model. It can be used to assess causality and stationarity, invertibility, and whether the AR and MA polynomials contain common roots.

The library also computes polynomial roots and inverse roots, making it possible to examine these properties numerically and graphically.

```python
diagnostics = arma.arma_diagnostics(
    ar=[0.6, -0.2],
    ma=[0.5],
)

print("Causal:", diagnostics["causal"])
print("Invertible:", diagnostics["invertible"])
```

See the accompanying documentation for a fuller discussion of the diagnostics and their interpretation.


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
\psi(L)=\frac{\theta(L)}{\varphi(L)}
       =\sum_{j=0}^{\infty}\psi_jL^j
$$

describes how a one-unit innovation propagates through the process. The library calculates the formal recursion

$$
\psi_0=1,
\qquad
\psi_j=\theta_j+\sum_{i=1}^{\min(p,j)}\varphi_i\psi_{j-i},
$$

where $\theta_j=0$ beyond the MA order.

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
=\frac{\theta(1)}{\varphi(1)}.
$$

The library reports this analytical long-run response only when the model is causal. Unit-root and explosive cases can still be inspected over a finite horizon, but no finite long-run response is asserted.


## Visualizations

ARMA Theory provides both **Matplotlib** and interactive **Plotly** visualizations for:

- theoretical ACF and PACF;
- dynamic responses ($\psi$-weights);
- cumulative responses;
- AR and MA inverse roots.

Inverse-root plots provide a graphical way to assess causality, invertibility, and possible common roots relative to the unit circle.

See the accompanying documentation for examples and the available plotting functions.



## Intended audience

ARMA Theory is intended for:

- undergraduate and postgraduate students in time series analysis or econometrics;
- instructors preparing lectures, demonstrations, and exercises;
- researchers who want transparent population-level ARMA calculations; and
- anyone interested in seeing how innovations propagate through a scalar dynamic model.


## Author

**Miguel A. Arranz**

Universidad Carlos III de Madrid

For questions, comments, or suggestions concerning ARMA Theory, please contact the author at [maarranz@eco.uc3m.es].


## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
