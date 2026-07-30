"""Diagnostics for theoretical ARMA models.

The coefficient convention used throughout this module is

    y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p}
          + epsilon_t + theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q}.

Thus the AR and MA polynomials, written in ascending powers of ``z``, are

    phi(z) = 1 - phi_1 z - ... - phi_p z**p
    theta(z) = 1 + theta_1 z + ... + theta_q z**q.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_coefficients(coefficients: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a validated one-dimensional array of finite real coefficients."""
    if isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Iterable):
        raise TypeError(f"{name} must be a one-dimensional iterable of real numbers.")

    values = list(coefficients)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
        raise TypeError(f"{name} coefficients must be real numbers.")

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} coefficients must all be finite.")
    return array


def _validate_root_tolerance(root_tolerance: float) -> float:
    """Return a validated common-root comparison tolerance."""
    if isinstance(root_tolerance, (bool, np.bool_)) or not isinstance(root_tolerance, Real):
        raise TypeError("root_tolerance must be a real number.")
    tolerance = float(root_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("root_tolerance must be finite and strictly positive.")
    return tolerance


def _polynomial_roots(polynomial: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Calculate roots from coefficients stored in ascending powers."""
    # np.roots expects descending powers and removes leading zeros. Reversing
    # here therefore also handles trailing zero AR or MA coefficients naturally.
    return np.asarray(np.roots(polynomial[::-1]), dtype=complex)


def _common_root_pairs(
    ar_roots: NDArray[np.complex128],
    ma_roots: NDArray[np.complex128],
    tolerance: float,
) -> list[dict[str, object]]:
    """Return AR/MA root pairs whose scaled distance is below ``tolerance``."""
    pairs: list[dict[str, object]] = []
    for ar_index, ar_root in enumerate(ar_roots):
        for ma_index, ma_root in enumerate(ma_roots):
            distance = abs(ar_root - ma_root)
            scale = max(1.0, abs(ar_root), abs(ma_root))
            scaled_distance = distance / scale
            if scaled_distance < tolerance:
                pairs.append(
                    {
                        "ar_index": ar_index,
                        "ma_index": ma_index,
                        "ar_root": ar_root,
                        "ma_root": ma_root,
                        "scaled_distance": float(scaled_distance),
                    }
                )
    return pairs


def arma_diagnostics(
    ar: ArrayLike,
    ma: ArrayLike,
    root_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Diagnose the polynomial structure of an ARMA model.

    Parameters
    ----------
    ar
        AR coefficients ``[phi_1, ..., phi_p]`` in the convention
        ``phi(z) = 1 - phi_1*z - ... - phi_p*z**p``. An empty iterable
        represents a pure MA model.
    ma
        MA coefficients ``[theta_1, ..., theta_q]`` in the convention
        ``theta(z) = 1 + theta_1*z + ... + theta_q*z**q``. An empty iterable
        represents a pure AR model.
    root_tolerance
        Strict upper bound for declaring an AR root and an MA root common.
        Roots ``a`` and ``b`` match when
        ``abs(a-b) / max(1, abs(a), abs(b)) < root_tolerance``.

    Returns
    -------
    dict
        Coefficients, ascending-power polynomials, roots, inverse roots,
        causality and invertibility flags, and common-root diagnostics.

    Notes
    -----
    A model is reported as causal (invertible) exactly when every AR (MA)
    polynomial root lies strictly outside the unit circle. Empty root sets
    satisfy their corresponding condition.
    """
    ar_array = _validate_coefficients(ar, "ar")
    ma_array = _validate_coefficients(ma, "ma")
    tolerance = _validate_root_tolerance(root_tolerance)

    ar_polynomial = np.concatenate(([1.0], -ar_array))
    ma_polynomial = np.concatenate(([1.0], ma_array))
    ar_roots = _polynomial_roots(ar_polynomial)
    ma_roots = _polynomial_roots(ma_polynomial)
    inverse_ar_roots = 1.0 / ar_roots
    inverse_ma_roots = 1.0 / ma_roots

    common_root_pairs = _common_root_pairs(ar_roots, ma_roots, tolerance)

    return {
        "order": (len(ar_array), len(ma_array)),
        "ar": ar_array,
        "ma": ma_array,
        "ar_polynomial": ar_polynomial,
        "ma_polynomial": ma_polynomial,
        "ar_roots": ar_roots,
        "ma_roots": ma_roots,
        "inverse_ar_roots": inverse_ar_roots,
        "inverse_ma_roots": inverse_ma_roots,
        "causal": bool(np.all(np.abs(ar_roots) > 1.0)),
        "invertible": bool(np.all(np.abs(ma_roots) > 1.0)),
        "common_roots": bool(common_root_pairs),
        "common_root_pairs": common_root_pairs,
        "minimal_representation": not common_root_pairs,
        "root_tolerance": tolerance,
    }


def arma_psi_weights(ar, ma, n_terms=20):
    """
    Compute the coefficients of the infinite MA representation of a causal
    ARMA model.

    The ARMA model is defined as

        y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p}
              + epsilon_t
              + theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q},

    or equivalently,

        phi(L) y_t = theta(L) epsilon_t,

    where

        phi(L) = 1 - phi_1 L - ... - phi_p L^p
        theta(L) = 1 + theta_1 L + ... + theta_q L^q.

    For a causal model,

        y_t = psi(L) epsilon_t
            = sum_{j=0}^infinity psi_j epsilon_{t-j}.

    The coefficients are computed recursively from

        psi_0 = 1

    and, for j >= 1,

        psi_j = theta_j
                + sum_{i=1}^{min(p, j)} phi_i psi_{j-i},

    where theta_j = 0 for j > q.

    Parameters
    ----------
    ar : array-like
        Autoregressive coefficients [phi_1, ..., phi_p].
        Use an empty list for a pure MA model.

    ma : array-like
        Moving-average coefficients [theta_1, ..., theta_q].
        Use an empty list for a pure AR model.

    n_terms : int, default=20
        Number of psi coefficients to return, including psi_0.

    Returns
    -------
    numpy.ndarray
        Array containing [psi_0, psi_1, ..., psi_{n_terms-1}].

    Raises
    ------
    ValueError
        If `n_terms` is not a positive integer or if the AR polynomial
        is not causal.

    Examples
    --------
    >>> arma_psi_weights(ar=[0.8], ma=[], n_terms=5)
    array([1.    , 0.8   , 0.64  , 0.512 , 0.4096])

    >>> arma_psi_weights(ar=[], ma=[0.5], n_terms=5)
    array([1. , 0.5, 0. , 0. , 0. ])
    """
    if not isinstance(n_terms, (int, np.integer)) or isinstance(n_terms, bool):
        raise ValueError("n_terms must be a positive integer.")

    if n_terms < 1:
        raise ValueError("n_terms must be at least 1.")

    ar = np.asarray(ar, dtype=float)
    ma = np.asarray(ma, dtype=float)

    diagnostics = arma_diagnostics(ar=ar, ma=ma)

    if not diagnostics["causal"]:
        raise ValueError(
            "The AR polynomial is not causal. "
            "A convergent infinite MA representation does not exist."
        )

    p = len(ar)
    q = len(ma)

    psi = np.zeros(n_terms, dtype=float)
    psi[0] = 1.0

    for j in range(1, n_terms):
        # theta_j is zero when j exceeds the MA order.
        theta_j = ma[j - 1] if j <= q else 0.0

        ar_component = 0.0

        for i in range(1, min(p, j) + 1):
            ar_component += ar[i - 1] * psi[j - i]

        psi[j] = theta_j + ar_component

    return psi


