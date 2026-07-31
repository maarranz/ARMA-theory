"""Theoretical tools for autoregressive moving-average (ARMA) models.

The module uses the convention

    y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p}
          + epsilon_t
          + theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q}.

Equivalently,

    phi(L) y_t = theta(L) epsilon_t,

where

    phi(L) = 1 - phi_1 L - ... - phi_p L^p
    theta(L) = 1 + theta_1 L + ... + theta_q L^q.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np


def _validate_coefficients(
    coefficients: Iterable[Real] | None,
    name: str,
) -> np.ndarray:
    """Return a validated one-dimensional array of finite coefficients."""
    if coefficients is None:
        return np.array([], dtype=float)

    if isinstance(coefficients, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence of numbers.")

    try:
        values = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a one-dimensional sequence of real numbers."
        ) from exc

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} coefficients must all be finite.")

    return values


def _polynomial_roots(coefficients: np.ndarray) -> np.ndarray:
    """Find roots when coefficients are ordered by ascending powers."""
    if coefficients.size <= 1:
        return np.array([], dtype=complex)

    # Remove zero coefficients at the highest powers. They do not affect the
    # polynomial and would otherwise cause NumPy to report spurious behaviour.
    last_nonzero = np.flatnonzero(coefficients != 0.0)
    if last_nonzero.size == 0 or last_nonzero[-1] == 0:
        return np.array([], dtype=complex)

    effective = coefficients[: last_nonzero[-1] + 1]
    return np.asarray(np.roots(effective[::-1]), dtype=complex)


def arma_diagnostics(
    ar: Iterable[Real] | None,
    ma: Iterable[Real] | None,
    root_tolerance: float = 1e-6,
) -> dict:
    """Diagnose the polynomial properties of an ARMA model.

    Parameters
    ----------
    ar
        Autoregressive coefficients ``[phi_1, ..., phi_p]``. Use an empty
        sequence for a pure MA model.
    ma
        Moving-average coefficients ``[theta_1, ..., theta_q]``. Use an empty
        sequence for a pure AR model.
    root_tolerance
        Positive relative tolerance used both for unit-circle classification
        and approximate common-root detection.

    Returns
    -------
    dict
        Model order, validated coefficients, polynomial coefficients, roots,
        inverse roots, causality, invertibility, and common-root diagnostics.

    Notes
    -----
    Polynomial coefficients are returned in ascending powers of the lag:
    ``[1, -phi_1, ..., -phi_p]`` and
    ``[1, theta_1, ..., theta_q]``.

    A root whose modulus is within ``root_tolerance`` of one is treated as
    lying on the unit circle, so it does not satisfy the strict causality or
    invertibility condition.
    """
    ar_values = _validate_coefficients(ar, "ar")
    ma_values = _validate_coefficients(ma, "ma")

    if (
        isinstance(root_tolerance, (bool, np.bool_))
        or not isinstance(root_tolerance, Real)
        or not np.isfinite(root_tolerance)
        or root_tolerance <= 0
    ):
        raise ValueError("root_tolerance must be a positive finite number.")

    root_tolerance = float(root_tolerance)
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
        ar_roots.size == 0
        or np.all(np.abs(ar_roots) > 1.0 + root_tolerance)
    )
    invertible = bool(
        ma_roots.size == 0
        or np.all(np.abs(ma_roots) > 1.0 + root_tolerance)
    )

    common_root_pairs = []
    for ar_index, ar_root in enumerate(ar_roots):
        for ma_index, ma_root in enumerate(ma_roots):
            scale = max(1.0, abs(ar_root), abs(ma_root))
            relative_distance = abs(ar_root - ma_root) / scale
            if relative_distance < root_tolerance:
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
        "root_tolerance": root_tolerance,
    }


def arma_psi_weights(
    ar: Iterable[Real] | None,
    ma: Iterable[Real] | None,
    n_terms: int = 20,
) -> np.ndarray:
    """Compute coefficients of a causal ARMA model's infinite MA form.

    For ``psi(L) = theta(L) / phi(L)``, the recursion is

    ``psi_0 = 1``

    and, for ``j >= 1``,

    ``psi_j = theta_j + sum(phi_i * psi_(j-i), i=1,...,min(p,j))``,

    where ``theta_j = 0`` when ``j`` exceeds the MA order.

    Parameters
    ----------
    ar
        Autoregressive coefficients ``[phi_1, ..., phi_p]``.
    ma
        Moving-average coefficients ``[theta_1, ..., theta_q]``.
    n_terms
        Number of weights to return, including ``psi_0``.

    Returns
    -------
    numpy.ndarray
        ``[psi_0, psi_1, ..., psi_(n_terms-1)]``.

    Raises
    ------
    ValueError
        If ``n_terms`` is invalid or the AR polynomial is not causal.
    """
    if (
        isinstance(n_terms, (bool, np.bool_))
        or not isinstance(n_terms, (int, np.integer))
        or n_terms < 1
    ):
        raise ValueError("n_terms must be a positive integer.")

    ar_values = _validate_coefficients(ar, "ar")
    ma_values = _validate_coefficients(ma, "ma")
    diagnostics = arma_diagnostics(ar_values, ma_values)

    if not diagnostics["causal"]:
        raise ValueError(
            "The AR polynomial is not causal; a convergent infinite MA "
            "representation does not exist."
        )

    psi = np.zeros(int(n_terms), dtype=float)
    psi[0] = 1.0

    p = ar_values.size
    q = ma_values.size

    for j in range(1, int(n_terms)):
        theta_j = ma_values[j - 1] if j <= q else 0.0
        upper_ar_lag = min(p, j)
        ar_component = sum(
            ar_values[i - 1] * psi[j - i]
            for i in range(1, upper_ar_lag + 1)
        )
        psi[j] = theta_j + ar_component

    return psi


def arma_autocovariances(
    ar,
    ma,
    n_lags=20,
    innovation_variance=1.0,
    n_psi=1000,
):
    """
    Compute theoretical autocovariances of a causal ARMA process.

    The autocovariances are approximated from the infinite MA
    representation:

        gamma_k = sigma^2 * sum_j psi_j * psi_{j+k}

    Parameters
    ----------
    ar : array-like
        AR coefficients [phi_1, ..., phi_p].

    ma : array-like
        MA coefficients [theta_1, ..., theta_q].

    n_lags : int, default=20
        Maximum autocovariance lag.

    innovation_variance : float, default=1.0
        Variance of the innovation process.

    n_psi : int, default=1000
        Number of psi-weights used in the approximation.

    Returns
    -------
    numpy.ndarray
        Autocovariances from lag 0 through `n_lags`.

    Raises
    ------
    ValueError
        If the arguments are invalid or the ARMA model is not causal.
    """
    import numpy as np

    if not isinstance(n_lags, int) or n_lags < 0:
        raise ValueError("n_lags must be a non-negative integer.")

    if not isinstance(n_psi, int) or n_psi <= n_lags:
        raise ValueError(
            "n_psi must be an integer greater than n_lags."
        )

    if innovation_variance <= 0:
        raise ValueError(
            "innovation_variance must be strictly positive."
        )

    psi = arma_psi_weights(
        ar=ar,
        ma=ma,
        n_terms=n_psi + n_lags,
    )

    gamma = np.empty(n_lags + 1, dtype=float)

    for lag in range(n_lags + 1):
        gamma[lag] = (
            innovation_variance
            * np.dot(
                psi[:n_psi],
                psi[lag : lag + n_psi],
            )
        )

    return gamma
def arma_acf(
    ar,
    ma,
    n_lags=20,
    n_psi=1000,
):
    """
    Compute the theoretical autocorrelation function of a causal ARMA process.

    Parameters
    ----------
    ar : array-like
        AR coefficients [phi_1, ..., phi_p].

    ma : array-like
        MA coefficients [theta_1, ..., theta_q].

    n_lags : int, default=20
        Maximum ACF lag.

    n_psi : int, default=1000
        Number of psi-weights used to approximate the autocovariances.

    Returns
    -------
    numpy.ndarray
        Theoretical autocorrelations from lag 0 through `n_lags`.
    """
    gamma = arma_autocovariances(
        ar=ar,
        ma=ma,
        n_lags=n_lags,
        innovation_variance=1.0,
        n_psi=n_psi,
    )

    if gamma[0] <= 0:
        raise ValueError(
            "The variance gamma[0] must be strictly positive."
        )

    return gamma / gamma[0]

def arma_pacf(
    ar=None,
    ma=None,
    acf=None,
    n_lags=20,
    n_psi=1000,
):
    """
    Compute the theoretical PACF using the Durbin-Levinson recursion.

    The function can either compute the ACF from ARMA coefficients or use
    a supplied autocorrelation sequence.

    Parameters
    ----------
    ar : array-like or None, default=None
        AR coefficients [phi_1, ..., phi_p].

    ma : array-like or None, default=None
        MA coefficients [theta_1, ..., theta_q].

    acf : array-like or None, default=None
        Precomputed autocorrelation sequence beginning with lag zero.
        If supplied, `ar` and `ma` are ignored.

    n_lags : int, default=20
        Maximum PACF lag.

    n_psi : int, default=1000
        Number of psi-weights used if the ACF must be computed.

    Returns
    -------
    numpy.ndarray
        Partial autocorrelations from lag 0 through `n_lags`.

        The value at lag zero is 1.

    Raises
    ------
    ValueError
        If the supplied arguments are invalid or the recursion fails.
    """
    import numpy as np

    if not isinstance(n_lags, int) or n_lags < 0:
        raise ValueError("n_lags must be a non-negative integer.")

    if acf is None:
        if ar is None:
            ar = []

        if ma is None:
            ma = []

        rho = arma_acf(
            ar=ar,
            ma=ma,
            n_lags=n_lags,
            n_psi=n_psi,
        )
    else:
        rho = np.asarray(acf, dtype=float)

        if rho.ndim != 1:
            raise ValueError("acf must be a one-dimensional sequence.")

        if len(rho) < n_lags + 1:
            raise ValueError(
                "acf must contain at least n_lags + 1 values."
            )

        rho = rho[: n_lags + 1]

        if not np.all(np.isfinite(rho)):
            raise ValueError("acf must contain only finite values.")

        if not np.isclose(rho[0], 1.0):
            raise ValueError("The ACF value at lag zero must equal 1.")

    pacf = np.empty(n_lags + 1, dtype=float)
    pacf[0] = 1.0

    if n_lags == 0:
        return pacf

    # phi_matrix[k, j] stores phi_{k,j}.
    phi_matrix = np.zeros(
        (n_lags + 1, n_lags + 1),
        dtype=float,
    )

    prediction_variance = np.empty(n_lags + 1, dtype=float)
    prediction_variance[0] = 1.0

    for k in range(1, n_lags + 1):
        numerator = rho[k]

        if k > 1:
            numerator -= np.dot(
                phi_matrix[k - 1, 1:k],
                rho[k - 1 : 0 : -1],
            )

        denominator = prediction_variance[k - 1]

        if denominator <= 0 or np.isclose(denominator, 0.0):
            raise ValueError(
                "Durbin-Levinson recursion encountered a "
                "non-positive prediction-error variance."
            )

        phi_kk = numerator / denominator
        phi_matrix[k, k] = phi_kk

        for j in range(1, k):
            phi_matrix[k, j] = (
                phi_matrix[k - 1, j]
                - phi_kk * phi_matrix[k - 1, k - j]
            )

        pacf[k] = phi_kk

        prediction_variance[k] = (
            prediction_variance[k - 1]
            * (1.0 - phi_kk**2)
        )

    return pacf



def plot_arma_acf(
    ar,
    ma,
    n_lags=20,
    n_psi=1000,
):
    """
    Plot the theoretical autocorrelation function of an ARMA model.

    Parameters
    ----------
    ar : array-like
        Autoregressive coefficients in the polynomial

            phi(L) = 1 - phi_1 L - ... - phi_p L^p.

    ma : array-like
        Moving-average coefficients in the polynomial

            theta(L) = 1 + theta_1 L + ... + theta_q L^q.

    n_lags : int, default=20
        Maximum lag to display. Lag zero is included.

    n_psi : int, default=1000
        Number of psi weights used to approximate the theoretical
        autocovariances.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.

    ax : matplotlib.axes.Axes
        Matplotlib axes containing the plot.

    Notes
    -----
    The graph shows a theoretical ACF. Therefore, confidence bands are
    not included.
    """
    import matplotlib.pyplot as plt

    if not isinstance(n_lags, int) or n_lags < 0:
        raise ValueError("n_lags must be a non-negative integer.")

    acf_values = arma_acf(
        ar=ar,
        ma=ma,
        n_lags=n_lags,
        n_psi=n_psi,
    )

    lags = np.arange(n_lags + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.vlines(
        lags,
        0,
        acf_values,
        linewidth=1.5,
    )

    ax.scatter(
        lags,
        acf_values,
        s=30,
        zorder=3,
    )

    ax.axhline(
        y=0,
        linewidth=1,
    )

    ax.set_title("Theoretical Autocorrelation Function")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    ax.set_xlim(-0.5, n_lags + 0.5)
    ax.set_ylim(
        min(-1.05, float(np.min(acf_values)) - 0.05),
        max(1.05, float(np.max(acf_values)) + 0.05),
    )

    ax.set_xticks(lags)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_arma_pacf(
    ar,
    ma,
    n_lags=20,
    n_psi=1000,
):
    """
    Plot the theoretical partial autocorrelation function of an ARMA model.

    Parameters
    ----------
    ar : array-like
        Autoregressive coefficients in the polynomial

            phi(L) = 1 - phi_1 L - ... - phi_p L^p.

    ma : array-like
        Moving-average coefficients in the polynomial

            theta(L) = 1 + theta_1 L + ... + theta_q L^q.

    n_lags : int, default=20
        Maximum lag to display. Lag zero is included.

    n_psi : int, default=1000
        Number of psi weights used to approximate the theoretical
        autocovariances and autocorrelations.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the plot.

    ax : matplotlib.axes.Axes
        Matplotlib axes containing the plot.

    Notes
    -----
    The graph shows a theoretical PACF. Therefore, confidence bands are
    not included.
    """
    import matplotlib.pyplot as plt

    if not isinstance(n_lags, int) or n_lags < 0:
        raise ValueError("n_lags must be a non-negative integer.")

    pacf_values = arma_pacf(
        ar=ar,
        ma=ma,
        n_lags=n_lags,
        n_psi=n_psi,
    )

    lags = np.arange(n_lags + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.vlines(
        lags,
        0,
        pacf_values,
        linewidth=1.5,
    )

    ax.scatter(
        lags,
        pacf_values,
        s=30,
        zorder=3,
    )

    ax.axhline(
        y=0,
        linewidth=1,
    )

    ax.set_title("Theoretical Partial Autocorrelation Function")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Partial Autocorrelation")

    ax.set_xlim(-0.5, n_lags + 0.5)
    ax.set_ylim(
        min(-1.05, float(np.min(pacf_values)) - 0.05),
        max(1.05, float(np.max(pacf_values)) + 0.05),
    )

    ax.set_xticks(lags)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return fig, ax

def plot_arma_correlations(
    ar,
    ma,
    n_lags=20,
    n_psi=1000,
):
    """
    Plot the theoretical ACF and PACF of an ARMA model together.

    Parameters
    ----------
    ar : array-like
        Autoregressive coefficients in the polynomial

            phi(L) = 1 - phi_1 L - ... - phi_p L^p.

    ma : array-like
        Moving-average coefficients in the polynomial

            theta(L) = 1 + theta_1 L + ... + theta_q L^q.

    n_lags : int, default=20
        Maximum lag to display. Lag zero is included.

    n_psi : int, default=1000
        Number of psi weights used to approximate the theoretical
        autocovariances and autocorrelations.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing both plots.

    axes : numpy.ndarray
        Array containing the ACF and PACF axes.

        ``axes[0]`` is the ACF plot and ``axes[1]`` is the PACF plot.

    Notes
    -----
    These are theoretical correlation functions. Confidence bands are
    therefore not included.
    """
    import matplotlib.pyplot as plt

    if not isinstance(n_lags, int) or n_lags < 0:
        raise ValueError("n_lags must be a non-negative integer.")

    acf_values = arma_acf(
        ar=ar,
        ma=ma,
        n_lags=n_lags,
        n_psi=n_psi,
    )

    pacf_values = arma_pacf(
        ar=ar,
        ma=ma,
        n_lags=n_lags,
        n_psi=n_psi,
    )

    lags = np.arange(n_lags + 1)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.5),
        sharex=True,
    )

    # Theoretical ACF
    axes[0].vlines(
        lags,
        0,
        acf_values,
        linewidth=1.5,
    )

    axes[0].scatter(
        lags,
        acf_values,
        s=30,
        zorder=3,
    )

    axes[0].axhline(
        y=0,
        linewidth=1,
    )

    axes[0].set_title("Theoretical ACF")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    axes[0].set_ylim(
        min(-1.05, float(np.min(acf_values)) - 0.05),
        max(1.05, float(np.max(acf_values)) + 0.05),
    )

    axes[0].grid(axis="y", alpha=0.3)

    # Theoretical PACF
    axes[1].vlines(
        lags,
        0,
        pacf_values,
        linewidth=1.5,
    )

    axes[1].scatter(
        lags,
        pacf_values,
        s=30,
        zorder=3,
    )

    axes[1].axhline(
        y=0,
        linewidth=1,
    )

    axes[1].set_title("Theoretical PACF")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial Autocorrelation")

    axes[1].set_ylim(
        min(-1.05, float(np.min(pacf_values)) - 0.05),
        max(1.05, float(np.max(pacf_values)) + 0.05),
    )

    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.set_xlim(-0.5, n_lags + 0.5)
        ax.set_xticks(lags)

    fig.tight_layout()

    return fig, axes


__all__ = [
    "arma_diagnostics",
    "arma_psi_weights",
    "arma_autocovariances",
    "arma_acf",
    "arma_pacf",
    "plot_arma_acf",
    "plot_arma_pacf",
    "plot_arma_correlations",
]
