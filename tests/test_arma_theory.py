"""Basic tests for the ARMA diagnostics layer."""

import unittest

import numpy as np

from arma_theory import arma_diagnostics


class TestArmaDiagnostics(unittest.TestCase):
    def test_reference_arma_model(self):
        result = arma_diagnostics(ar=[0.6, -0.2], ma=[0.5])

        self.assertEqual(result["order"], (2, 1))
        np.testing.assert_allclose(result["ar_polynomial"], [1.0, -0.6, 0.2])
        np.testing.assert_allclose(result["ma_polynomial"], [1.0, 0.5])
        np.testing.assert_allclose(
            np.sort_complex(result["ar_roots"]),
            np.sort_complex([1.5 + 1.658312395j, 1.5 - 1.658312395j]),
        )
        np.testing.assert_allclose(result["ma_roots"], [-2.0])
        self.assertTrue(result["causal"])
        self.assertTrue(result["invertible"])
        self.assertFalse(result["common_roots"])
        self.assertTrue(result["minimal_representation"])

    def test_empty_components(self):
        pure_ma = arma_diagnostics([], [0.5])
        pure_ar = arma_diagnostics([0.5], [])

        self.assertTrue(pure_ma["causal"])
        self.assertEqual(pure_ma["ar_roots"].size, 0)
        self.assertTrue(pure_ar["invertible"])
        self.assertEqual(pure_ar["ma_roots"].size, 0)

    def test_noncausal_and_noninvertible_models(self):
        self.assertFalse(arma_diagnostics([1.2], [])["causal"])
        self.assertFalse(arma_diagnostics([], [2.0])["invertible"])

    def test_scaled_common_root_detection(self):
        # phi(z) = 1 - 0.5z and theta(z) = 1 - 0.5z share z = 2.
        result = arma_diagnostics([0.5], [-0.5])

        self.assertTrue(result["common_roots"])
        self.assertFalse(result["minimal_representation"])
        self.assertEqual(len(result["common_root_pairs"]), 1)
        self.assertAlmostEqual(
            result["common_root_pairs"][0]["scaled_distance"], 0.0
        )

    def test_invalid_inputs(self):
        invalid_cases = [
            ((0.5), [], 1e-6, TypeError),
            ([True], [], 1e-6, TypeError),
            ([np.nan], [], 1e-6, ValueError),
            ([0.5], [], 0.0, ValueError),
            ([0.5], [], "small", TypeError),
        ]
        for ar, ma, tolerance, error in invalid_cases:
            with self.subTest(ar=ar, ma=ma, tolerance=tolerance):
                with self.assertRaises(error):
                    arma_diagnostics(ar, ma, tolerance)


if __name__ == "__main__":
    unittest.main()
