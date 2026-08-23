"""Theoretical analysis and visualization tools for ARMA models.

The module uses the convention

    phi(L) y_t = theta(L) epsilon_t,

where ``phi(L) = 1 - phi_1 L - ... - phi_p L**p`` and
``theta(L) = 1 + theta_1 L + ... + theta_q L**q``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _validate_coefficients(
    coefficients: Iterable[Real] | None, name: str
) -> np.ndarray:
    """Return coefficients as a finite, one-dimensional float array."""
    if coefficients is None:
        return np.array([], dtype=float)
    if isinstance(coefficients, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence of numbers.")
    try:
        values = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of real numbers.") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} coefficients must all be finite.")
    return values


def _validate_nonnegative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _polynomial_roots(coefficients: np.ndarray) -> np.ndarray:
    """Find polynomial roots when coefficients use ascending powers."""
    nonzero = np.flatnonzero(coefficients != 0.0)
    if nonzero.size == 0 or nonzero[-1] == 0:
        return np.array([], dtype=complex)
    effective = coefficients[: nonzero[-1] + 1]
    return np.asarray(np.roots(effective[::-1]), dtype=complex)


def arma_diagnostics(
    ar: Iterable[Real] | None,
    ma: Iterable[Real] | None,
    root_tolerance: float = 1e-6,
) -> dict:
    """Return roots and diagnostic properties of an ARMA specification."""
    ar_values = _validate_coefficients(ar, "ar")
    ma_values = _validate_coefficients(ma, "ma")
    if (
        isinstance(root_tolerance, (bool, np.bool_))
        or not isinstance(root_tolerance, Real)
        or not np.isfinite(root_tolerance)
        or root_tolerance <= 0
    ):
        raise ValueError("root_tolerance must be a positive finite number.")
    tolerance = float(root_tolerance)
    ar_polynomial = np.concatenate(([1.0], -ar_values))
    ma_polynomial = np.concatenate(([1.0], ma_values))
    ar_roots = _polynomial_roots(ar_polynomial)
    ma_roots = _polynomial_roots(ma_polynomial)
    inverse_ar_roots = (
        1.0 / ar_roots if ar_roots.size else np.array([], dtype=complex)
    )
    inverse_ma_roots = (
        1.0 / ma_roots if ma_roots.size else np.array([], dtype=complex)
    )
    causal = bool(
        ar_roots.size == 0 or np.all(np.abs(ar_roots) > 1.0 + tolerance)
    )
    invertible = bool(
        ma_roots.size == 0 or np.all(np.abs(ma_roots) > 1.0 + tolerance)
    )
    common_root_pairs = []
    for ar_index, ar_root in enumerate(ar_roots):
        for ma_index, ma_root in enumerate(ma_roots):
            scale = max(1.0, abs(ar_root), abs(ma_root))
            relative_distance = abs(ar_root - ma_root) / scale
            if relative_distance <= tolerance:
                common_root_pairs.append(
                    {
                        "ar_index": ar_index,
                        "ma_index": ma_index,
                        "ar_root": ar_root,
                        "ma_root": ma_root,
                        "relative_distance": float(relative_distance),
                    }
                )
    common_roots = bool(common_root_pairs)
    return {
        "order": (ar_values.size, ma_values.size),
        "ar": ar_values.copy(),
        "ma": ma_values.copy(),
        "ar_polynomial": ar_polynomial,
        "ma_polynomial": ma_polynomial,
        "ar_roots": ar_roots,
        "ma_roots": ma_roots,
        "inverse_ar_roots": inverse_ar_roots,
        "inverse_ma_roots": inverse_ma_roots,
        "causal": causal,
        "stationary": causal,
        "invertible": invertible,
        "common_roots": common_roots,
        "common_root_pairs": common_root_pairs,
        "minimal_representation": not common_roots,
        "root_tolerance": tolerance,
    }


def arma_psi_weights(
    ar: Iterable[Real] | None,
    ma: Iterable[Real] | None,
    n_terms: int = 20,
) -> np.ndarray:
    """Compute formal psi weights for any finite ARMA coefficients."""
    if (
        isinstance(n_terms, (bool, np.bool_))
        or not isinstance(n_terms, (int, np.integer))
        or n_terms < 1
    ):
        raise ValueError("n_terms must be a positive integer.")
    ar_values = _validate_coefficients(ar, "ar")
    ma_values = _validate_coefficients(ma, "ma")
    psi = np.zeros(int(n_terms), dtype=float)
    psi[0] = 1.0
    for j in range(1, int(n_terms)):
        theta_j = ma_values[j - 1] if j <= ma_values.size else 0.0
        upper = min(ar_values.size, j)
        psi[j] = theta_j + np.dot(ar_values[:upper], psi[j - 1 :: -1][:upper])
    return psi


def arma_autocovariances(
    ar: Iterable[Real] | None,
    ma: Iterable[Real] | None,
    n_lags: int = 20,
    innovation_variance: float = 1.0,
    n_psi: int = 1000,
) -> np.ndarray:
    """Compute theoretical autocovariances of a causal ARMA process."""
    n_lags = _validate_nonnegative_integer(n_lags, "n_lags")
    if (
        isinstance(n_psi, (bool, np.bool_))
        or not isinstance(n_psi, (int, np.integer))
        or n_psi <= n_lags
    ):
        raise ValueError("n_psi must be an integer greater than n_lags.")
    if (
        isinstance(innovation_variance, (bool, np.bool_))
        or not isinstance(innovation_variance, Real)
        or not np.isfinite(innovation_variance)
        or innovation_variance <= 0
    ):
        raise ValueError("innovation_variance must be strictly positive and finite.")
    diagnostics = arma_diagnostics(ar, ma)
    if not diagnostics["causal"]:
        raise ValueError("Theoretical autocovariances require a causal ARMA model.")
    psi = arma_psi_weights(ar, ma, int(n_psi) + n_lags)
    return float(innovation_variance) * np.array(
        [np.dot(psi[: int(n_psi)], psi[k : k + int(n_psi)]) for k in range(n_lags + 1)]
    )


def arma_acf(
    ar: Iterable[Real] | None,
    ma: Iterable[Real] | None,
    n_lags: int = 20,
    n_psi: int = 1000,
) -> np.ndarray:
    """Compute the theoretical ACF of a causal ARMA process."""
    gamma = arma_autocovariances(ar, ma, n_lags, 1.0, n_psi)
    if not np.isfinite(gamma[0]) or gamma[0] <= 0:
        raise ValueError("The variance gamma[0] must be strictly positive.")
    return gamma / gamma[0]


def arma_pacf(
    ar: Iterable[Real] | None = None,
    ma: Iterable[Real] | None = None,
    acf: Iterable[Real] | None = None,
    n_lags: int = 20,
    n_psi: int = 1000,
) -> np.ndarray:
    """Compute theoretical PACF values with the Durbin-Levinson recursion."""
    n_lags = _validate_nonnegative_integer(n_lags, "n_lags")
    if acf is None:
        rho = arma_acf(ar, ma, n_lags, n_psi)
    else:
        rho = np.asarray(acf, dtype=float)
        if rho.ndim != 1:
            raise ValueError("acf must be a one-dimensional sequence.")
        if rho.size < n_lags + 1:
            raise ValueError("acf must contain at least n_lags + 1 values.")
        rho = rho[: n_lags + 1]
        if not np.all(np.isfinite(rho)):
            raise ValueError("acf must contain only finite values.")
        if not np.isclose(rho[0], 1.0):
            raise ValueError("The ACF value at lag zero must equal 1.")
    pacf = np.empty(n_lags + 1, dtype=float)
    pacf[0] = 1.0
    if n_lags == 0:
        return pacf
    phi = np.zeros((n_lags + 1, n_lags + 1), dtype=float)
    variance = np.empty(n_lags + 1, dtype=float)
    variance[0] = 1.0
    for k in range(1, n_lags + 1):
        numerator = rho[k]
        if k > 1:
            numerator -= np.dot(phi[k - 1, 1:k], rho[k - 1 : 0 : -1])
        if variance[k - 1] <= 0 or np.isclose(variance[k - 1], 0.0):
            raise ValueError(
                "Durbin-Levinson recursion encountered a non-positive "
                "prediction-error variance."
            )
        phi[k, k] = numerator / variance[k - 1]
        if k > 1:
            phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, k - 1 : 0 : -1]
        pacf[k] = phi[k, k]
        variance[k] = variance[k - 1] * (1.0 - phi[k, k] ** 2)
    return pacf


def _correlation_axis(ax, lags: np.ndarray, values: np.ndarray, title: str, ylabel: str) -> None:
    ax.vlines(lags, 0, values, linewidth=1.5)
    ax.scatter(lags, values, s=30, zorder=3)
    ax.axhline(0, linewidth=1)
    ax.set(title=title, xlabel="Lag", ylabel=ylabel)
    ax.set_xlim(-0.5, lags[-1] + 0.5)
    ax.set_ylim(min(-1.05, float(values.min()) - 0.05), max(1.05, float(values.max()) + 0.05))
    ax.set_xticks(lags)
    ax.grid(axis="y", alpha=0.3)


def plot_arma_acf(ar, ma, n_lags: int = 20, n_psi: int = 1000):
    """Plot the theoretical ACF; return ``(figure, axes)``."""
    values = arma_acf(ar, ma, n_lags, n_psi)
    lags = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _correlation_axis(ax, lags, values, "Theoretical Autocorrelation Function", "Autocorrelation")
    fig.tight_layout()
    return fig, ax


def plot_arma_pacf(ar, ma, n_lags: int = 20, n_psi: int = 1000):
    """Plot the theoretical PACF; return ``(figure, axes)``."""
    values = arma_pacf(ar, ma, n_lags=n_lags, n_psi=n_psi)
    lags = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _correlation_axis(ax, lags, values, "Theoretical Partial Autocorrelation Function", "Partial Autocorrelation")
    fig.tight_layout()
    return fig, ax


def plot_arma_correlations(ar, ma, n_lags: int = 20, n_psi: int = 1000):
    """Plot theoretical ACF and PACF side by side."""
    acf_values = arma_acf(ar, ma, n_lags, n_psi)
    pacf_values = arma_pacf(ar, ma, n_lags=n_lags, n_psi=n_psi)
    lags = np.arange(acf_values.size)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    _correlation_axis(axes[0], lags, acf_values, "Theoretical ACF", "Autocorrelation")
    _correlation_axis(axes[1], lags, pacf_values, "Theoretical PACF", "Partial Autocorrelation")
    fig.tight_layout()
    return fig, axes


def arma_dynamics(ar, ma, horizon: int = 20) -> dict:
    """Return impulse-response psi weights and their cumulative response."""
    horizon = _validate_nonnegative_integer(horizon, "horizon")
    ar_values = _validate_coefficients(ar, "ar")
    ma_values = _validate_coefficients(ma, "ma")
    diagnostics = arma_diagnostics(ar_values, ma_values)
    psi = arma_psi_weights(ar_values, ma_values, horizon + 1)
    causal = diagnostics["causal"]
    long_run_response = None
    if causal:
        long_run_response = (1.0 + ma_values.sum()) / (1.0 - ar_values.sum())
    return {
        "psi": psi,
        "cumulative_psi": np.cumsum(psi),
        "causal": causal,
        "long_run_exists": causal,
        "long_run_response": long_run_response,
        "horizon": horizon,
    }


def plot_arma_dynamic_response(ar, ma, horizon: int = 20):
    """Plot the impulse-response psi weights."""
    results = arma_dynamics(ar, ma, horizon)
    x = np.arange(results["horizon"] + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.vlines(x, 0, results["psi"], linewidth=1.5)
    ax.scatter(x, results["psi"], s=30, zorder=3)
    ax.axhline(0, linewidth=1)
    ax.set(title="Dynamic Response to a Unit Innovation", xlabel="Horizon", ylabel=r"$\psi_j$")
    ax.set_xlim(-0.5, results["horizon"] + 0.5)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_arma_cumulative_response(ar, ma, horizon: int = 20):
    """Plot the cumulative impulse response."""
    results = arma_dynamics(ar, ma, horizon)
    x = np.arange(results["horizon"] + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, results["cumulative_psi"], marker="o")
    if results["long_run_exists"]:
        ax.axhline(results["long_run_response"], linestyle="--", linewidth=1.2, label="Long-run response")
        ax.legend()
    ax.set(title="Cumulative Response to a Unit Innovation", xlabel="Horizon", ylabel=r"$\sum_{i=0}^{j}\psi_i$")
    ax.set_xlim(-0.5, results["horizon"] + 0.5)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def arma_summary(results: Mapping) -> None:
    """Print a compact table for results returned by :func:`arma_dynamics`."""
    required = {"psi", "cumulative_psi", "causal", "long_run_exists", "long_run_response", "horizon"}
    missing = required.difference(results)
    if missing:
        raise ValueError("results is missing required keys: " + ", ".join(sorted(missing)))
    psi = np.asarray(results["psi"], dtype=float)
    cumulative = np.asarray(results["cumulative_psi"], dtype=float)
    if psi.ndim != 1 or cumulative.ndim != 1 or psi.size != cumulative.size:
        raise ValueError("psi and cumulative_psi must be one-dimensional and have the same length.")
    print("ARMA Dynamic Analysis")
    print("=" * 42)
    print(f"Causal model ............. {results['causal']}")
    long_run = f"{results['long_run_response']:.6f}" if results["long_run_exists"] else "Not defined"
    print(f"Long-run response ........ {long_run}")
    print(f"Analysis horizon ......... {results['horizon']}")
    print(f"\n{'Horizon':>8} {'psi_j':>12} {'Cumulative psi':>16}")
    print("-" * 38)
    for j, (psi_j, cumulative_j) in enumerate(zip(psi, cumulative)):
        print(f"{j:8d} {psi_j:12.4f} {cumulative_j:16.4f}")


def _inverse_root_data(ar, ma, root_tolerance):
    diagnostics = arma_diagnostics(ar, ma, root_tolerance)
    ar_inv = np.asarray(diagnostics["inverse_ar_roots"], dtype=complex)
    ma_inv = np.asarray(diagnostics["inverse_ma_roots"], dtype=complex)
    common_inv = np.asarray(
        [1.0 / pair["ar_root"] for pair in diagnostics["common_root_pairs"]], dtype=complex
    )
    all_roots = np.concatenate((ar_inv, ma_inv))
    limit = 1.15 * max(1.0, float(np.max(np.abs(all_roots))) if all_roots.size else 1.0)
    return diagnostics, ar_inv, ma_inv, common_inv, limit


def plot_arma_inverse_roots(ar, ma, root_tolerance: float = 1e-6):
    """Plot inverse AR and MA roots in the complex plane with Matplotlib."""
    _, ar_inv, ma_inv, common_inv, limit = _inverse_root_data(ar, ma, root_tolerance)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    angles = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(angles), np.sin(angles), "--", linewidth=1.2, label="Unit circle")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    if ar_inv.size:
        ax.scatter(ar_inv.real, ar_inv.imag, marker="x", s=80, linewidths=2, label="AR inverse roots", zorder=3)
    if ma_inv.size:
        ax.scatter(ma_inv.real, ma_inv.imag, marker="o", s=70, facecolors="none", linewidths=1.8, label="MA inverse roots", zorder=3)
    if common_inv.size:
        ax.scatter(common_inv.real, common_inv.imag, marker="+", s=180, linewidths=2.5, label="Common inverse root", zorder=5)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), title="Inverse Roots of AR and MA Polynomials", xlabel="Real part", ylabel="Imaginary part")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plotly_arma_inverse_roots(ar, ma, root_tolerance: float = 1e-6):
    """Plot inverse AR and MA roots interactively with Plotly."""
    _, ar_inv, ma_inv, common_inv, limit = _inverse_root_data(ar, ma, root_tolerance)
    angles = np.linspace(0, 2 * np.pi, 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(angles), y=np.sin(angles), mode="lines", name="Unit circle", line={"dash": "dash"}, hoverinfo="skip"))
    if ar_inv.size:
        fig.add_trace(go.Scatter(x=ar_inv.real, y=ar_inv.imag, mode="markers", name="AR inverse roots", marker={"symbol": "x", "size": 12}, customdata=np.abs(ar_inv), hovertemplate="Real: %{x:.4f}<br>Imaginary: %{y:.4f}<br>Modulus: %{customdata:.4f}<extra></extra>"))
    if ma_inv.size:
        fig.add_trace(go.Scatter(x=ma_inv.real, y=ma_inv.imag, mode="markers", name="MA inverse roots", marker={"symbol": "circle", "size": 11}, customdata=np.abs(ma_inv), hovertemplate="Real: %{x:.4f}<br>Imaginary: %{y:.4f}<br>Modulus: %{customdata:.4f}<extra></extra>"))
    if common_inv.size:
        fig.add_trace(go.Scatter(x=common_inv.real, y=common_inv.imag, mode="markers", name="Common inverse root", marker={"symbol": "cross", "size": 16}, hovertemplate="Common root<br>Real: %{x:.4f}<br>Imaginary: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)
    fig.update_xaxes(title="Real part", range=[-limit, limit], constrain="domain")
    fig.update_yaxes(title="Imaginary part", range=[-limit, limit], scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Inverse Roots of AR and MA Polynomials", width=700, height=700)
    return fig


def _add_plotly_stems(fig, lags, values, name, row=None, col=None):
    for lag, value in zip(lags, values):
        fig.add_shape(type="line", x0=lag, x1=lag, y0=0, y1=value, line={"width": 1.5}, row=row, col=col)
    kwargs = {} if row is None else {"row": row, "col": col}
    fig.add_trace(go.Scatter(x=lags, y=values, mode="markers", name=name, marker={"size": 8}, text=[f"Lag: {lag}<br>{name}: {value:.4f}" for lag, value in zip(lags, values)], hovertemplate="%{text}<extra></extra>", showlegend=False), **kwargs)


def _plotly_correlation(ar, ma, n_lags, n_psi, kind):
    values = arma_acf(ar, ma, n_lags, n_psi) if kind == "ACF" else arma_pacf(ar, ma, n_lags=n_lags, n_psi=n_psi)
    lags = np.arange(values.size)
    fig = go.Figure()
    _add_plotly_stems(fig, lags, values, kind)
    fig.add_hline(y=0, line_width=1)
    title = "Theoretical Autocorrelation Function" if kind == "ACF" else "Theoretical Partial Autocorrelation Function"
    ylabel = "Autocorrelation" if kind == "ACF" else "Partial Autocorrelation"
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title=ylabel, showlegend=False)
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=1, range=[-0.5, n_lags + 0.5])
    fig.update_yaxes(range=[min(-1.05, float(values.min()) - 0.05), max(1.05, float(values.max()) + 0.05)])
    return fig


def plotly_arma_acf(ar, ma, n_lags: int = 20, n_psi: int = 1000):
    """Plot the theoretical ACF interactively."""
    return _plotly_correlation(ar, ma, n_lags, n_psi, "ACF")


def plotly_arma_pacf(ar, ma, n_lags: int = 20, n_psi: int = 1000):
    """Plot the theoretical PACF interactively."""
    return _plotly_correlation(ar, ma, n_lags, n_psi, "PACF")


def plotly_arma_correlations(ar, ma, n_lags: int = 20, n_psi: int = 1000):
    """Plot theoretical ACF and PACF side by side with Plotly."""
    acf_values = arma_acf(ar, ma, n_lags, n_psi)
    pacf_values = arma_pacf(ar, ma, n_lags=n_lags, n_psi=n_psi)
    lags = np.arange(acf_values.size)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Theoretical ACF", "Theoretical PACF"))
    _add_plotly_stems(fig, lags, acf_values, "ACF", 1, 1)
    _add_plotly_stems(fig, lags, pacf_values, "PACF", 1, 2)
    for col in (1, 2):
        fig.add_hline(y=0, line_width=1, row=1, col=col)
        fig.update_xaxes(title_text="Lag", tickmode="linear", tick0=0, dtick=1, range=[-0.5, n_lags + 0.5], row=1, col=col)
    fig.update_yaxes(title_text="Autocorrelation", range=[min(-1.05, float(acf_values.min()) - 0.05), max(1.05, float(acf_values.max()) + 0.05)], row=1, col=1)
    fig.update_yaxes(title_text="Partial Autocorrelation", range=[min(-1.05, float(pacf_values.min()) - 0.05), max(1.05, float(pacf_values.max()) + 0.05)], row=1, col=2)
    fig.update_layout(title="Theoretical ACF and PACF", width=1100, height=450)
    return fig


def plotly_arma_dynamic_response(
    ar,
    ma,
    horizon=20,
):
    """
    Plot the psi-weights of an ARMA model interactively with Plotly.

    The psi-weights show the dynamic response of the process to a
    one-unit innovation.

    Parameters
    ----------
    ar : array-like
        Autoregressive coefficients.

    ma : array-like
        Moving-average coefficients.

    horizon : int, default=20
        Maximum response horizon.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    results = arma_dynamics(
        ar=ar,
        ma=ma,
        horizon=horizon,
    )

    psi = results["psi"]
    horizons = np.arange(horizon + 1)

    fig = go.Figure()

    # Vertical response lines
    for h, value in zip(horizons, psi):
        fig.add_shape(
            type="line",
            x0=h,
            x1=h,
            y0=0,
            y1=value,
            line=dict(width=1.5),
        )

    hover_text = [
        (
            f"Horizon: {h}<br>"
            f"psi: {value:.4f}"
        )
        for h, value in zip(horizons, psi)
    ]

    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=psi,
            mode="markers",
            name="Dynamic response",
            marker=dict(size=8),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    fig.update_layout(
        title="Dynamic Response to a Unit Innovation",
        xaxis_title="Horizon",
        yaxis_title="psi",
        showlegend=False,
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        range=[-0.5, horizon + 0.5],
    )

    return fig


def plotly_arma_cumulative_response(
    ar,
    ma,
    horizon=20,
):
    """
    Plot the cumulative psi-response of an ARMA model interactively.

    Parameters
    ----------
    ar : array-like
        Autoregressive coefficients.

    ma : array-like
        Moving-average coefficients.

    horizon : int, default=20
        Maximum response horizon.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    results = arma_dynamics(
        ar=ar,
        ma=ma,
        horizon=horizon,
    )

    cumulative = results["cumulative_psi"]
    psi = results["psi"]
    horizons = np.arange(horizon + 1)

    hover_text = [
        (
            f"Horizon: {h}<br>"
            f"psi: {psi_j:.4f}<br>"
            f"Cumulative response: {cum_j:.4f}"
        )
        for h, psi_j, cum_j in zip(
            horizons,
            psi,
            cumulative,
        )
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative response",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add analytical LR response only for causal models.
    if results["long_run_exists"]:
        fig.add_hline(
            y=results["long_run_response"],
            line_dash="dash",
            line_width=1.5,
            annotation_text=(
                f"Long-run response = "
                f"{results['long_run_response']:.4f}"
            ),
            annotation_position="top right",
        )

    fig.add_hline(
        y=0,
        line_width=1,
    )

    fig.update_layout(
        title="Cumulative Response to a Unit Innovation",
        xaxis_title="Horizon",
        yaxis_title="Cumulative psi",
        showlegend=False,
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        range=[-0.5, horizon + 0.5],
    )

    return fig


__all__ = [
    "arma_diagnostics",
    "arma_psi_weights",
    "arma_autocovariances",
    "arma_acf",
    "arma_pacf",
    "plot_arma_acf",
    "plot_arma_pacf",
    "plot_arma_correlations",
    "arma_dynamics",
    "plot_arma_dynamic_response",
    "plot_arma_cumulative_response",
    "arma_summary",
    "plot_arma_inverse_roots",
    "plotly_arma_inverse_roots",
    "plotly_arma_acf",
    "plotly_arma_pacf",
    "plotly_arma_correlations",
    "plotly_arma_dynamic_response",
    "plotly_arma_cumulative_response",
]
