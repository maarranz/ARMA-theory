# ARMA Theory

**ARMA Theory** is an educational Python project for exploring the theoretical
properties of autoregressive moving-average (ARMA) models.

It is intended for students and instructors in time series analysis,
econometrics, and related fields. Rather than estimating models from observed
data, the project shows how specified AR and MA coefficients determine the
mathematical behaviour of an ARMA process through numerical results and
visualizations.

The repository includes the companion notebook
`ARMA_Theory_Demo.ipynb`, which presents worked examples using the
`arma_theory` module.

## Project goals

The project is designed to support the theoretical analysis of ARMA models,
including:

- causality and stationarity;
- invertibility;
- characteristic AR and MA roots;
- common-root detection and minimal representations;
- the infinite moving-average representation (ψ-weights);
- theoretical autocovariances;
- the theoretical autocorrelation function (ACF);
- the theoretical partial autocorrelation function (PACF); and
- graphical representations of correlations and roots.

ARMA Theory is deliberately focused on theory. Estimation, forecasting, and
empirical analysis of observed data are outside the scope of the project.

## Model convention

Throughout the project, an ARMA\((p,q)\) model is written as

$$
y_t
=
\phi_1 y_{t-1}
+ \cdots
+ \phi_p y_{t-p}
+ \varepsilon_t
+ \theta_1 \varepsilon_{t-1}
+ \cdots
+ \theta_q \varepsilon_{t-q}.
$$

Equivalently,

$$
\phi(L)y_t=\theta(L)\varepsilon_t,
$$

where

$$
\phi(L)=1-\phi_1L-\cdots-\phi_pL^p
$$

and

$$
\theta(L)=1+\theta_1L+\cdots+\theta_qL^q.
$$

This sign convention is used consistently in the library, notebook, and
documentation.

For example,

```python
ar = [0.6, -0.2]
ma = [0.5]
```

represents

$$
y_t
=0.6y_{t-1}
-0.2y_{t-2}
+\varepsilon_t
+0.5\varepsilon_{t-1}.
$$

Use an empty list for a component that is absent:

```python
ar = []  # Pure MA model
ma = []  # Pure AR model
```

## Current features

The current development version includes:

- construction of AR and MA polynomials;
- calculation of characteristic and inverse roots;
- causality checks for the AR component;
- invertibility checks for the MA component;
- approximate common-root detection;
- identification of nonminimal ARMA representations; and
- calculation of ψ-weights for the infinite MA representation.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/maarranz/ARMA-theory.git
cd ARMA-theory
```

### 2. Create the environment

Create the project environment from `environment.yml` using Mamba:

```bash
mamba env create -f environment.yml
```

### 3. Activate the environment

```bash
mamba activate arma_theory
```

### 4. Launch the demonstration notebook

```bash
jupyter lab
```

Then open `ARMA_Theory_Demo.ipynb`.

To use the module directly from the repository:

```python
import arma_theory as arma

results = arma.arma_diagnostics(
    ar=[0.6, -0.2],
    ma=[0.5],
)
```

## Repository structure

```text
ARMA-theory/
├── arma_theory.py          # Core theoretical calculations
├── ARMA_Theory_Demo.ipynb  # Worked examples and visualizations
├── environment.yml         # Mamba/Conda environment specification
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

The structure may evolve as tests, examples, and additional teaching materials
are added.

## Roadmap

Planned additions include:

- theoretical autocovariances;
- theoretical ACF values and plots;
- theoretical PACF values and plots;
- automatic convergence control for the infinite MA representation;
- AR and MA inverse-root diagrams;
- combined numerical summary tables;
- automated tests for the mathematical components; and
- additional worked examples in the teaching notebook.

The theoretical PACF will be calculated from population autocorrelations using
the Durbin–Levinson recursion. This design choice applies specifically to the
theoretical PACF and does not prescribe a method for estimating the PACF from
data.

## Intended audience

ARMA Theory is designed primarily for:

- undergraduate and postgraduate students;
- instructors teaching time series analysis or econometrics;
- researchers seeking transparent theoretical ARMA calculations; and
- anyone interested in understanding how ARMA coefficients shape the
  properties of a stochastic process.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for
details.
