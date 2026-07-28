import hashlib
import io
import json
import unittest
from pathlib import Path


from src.single_token_complementarity import (
    PINNED_SINGLE_TOKEN_MATRIX_SHA256,
    align_fingerprint_pairs,
    read_distance_matrix,
    read_verified_distance_matrix,
    render_latex_table,
    summarize_alignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SingleTokenComplementarityTests(unittest.TestCase):
    def test_reads_upper_triangle_from_symmetric_distance_csv(self):
        source = io.StringIO(
            'model,"vendor/a","vendor/b","vendor/c"\n'
            '"vendor/a",0,0.1,0.5\n'
            '"vendor/b",0.1,0,0.3\n'
            '"vendor/c",0.5,0.3,0\n'
        )

        distances = read_distance_matrix(source)

        self.assertEqual(
            distances,
            {
                ("vendor/a", "vendor/b"): 0.1,
                ("vendor/a", "vendor/c"): 0.5,
                ("vendor/b", "vendor/c"): 0.3,
            },
        )

    def test_aligns_only_exact_non_thinking_model_ids(self):
        knowledge_pairs = {
            "a||b": {"jaccard": 0.9, "hss": 0.8, "lift": 2.0, "both_wrong": 20},
            "a-think||b": {"jaccard": 1.0, "hss": 1.0, "lift": 3.0, "both_wrong": 20},
            "a||missing": {"jaccard": 0.2, "hss": 0.1, "lift": 1.0, "both_wrong": 20},
        }
        model_ids = {
            "a": "vendor/a",
            "a-think": "vendor/a",
            "b": "vendor/b",
            "missing": "vendor/missing",
        }
        distances = {("vendor/a", "vendor/b"): 0.1}

        aligned = align_fingerprint_pairs(knowledge_pairs, model_ids, distances)

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]["model_a"], "vendor/a")
        self.assertEqual(aligned[0]["model_b"], "vendor/b")
        self.assertEqual(aligned[0]["jsd"], 0.1)

    def test_summarizes_complementarity_and_hss_support_filter(self):
        aligned = [
            {"model_a": "v/a", "model_b": "v/b", "jsd": 0.1,
             "jaccard": 0.9, "hss": 0.8, "lift": 2.0, "both_wrong": 20},
            {"model_a": "v/a", "model_b": "w/c", "jsd": 0.5,
             "jaccard": 0.2, "hss": 0.1, "lift": 1.0, "both_wrong": 20},
            {"model_a": "v/b", "model_b": "w/c", "jsd": 0.3,
             "jaccard": 0.4, "hss": 0.5, "lift": 1.5, "both_wrong": 5},
        ]

        summary = summarize_alignment(aligned, min_joint_wrong=10)

        self.assertEqual(summary["n_models"], 3)
        self.assertEqual(summary["n_pairs"], 3)
        self.assertEqual(summary["correlations"]["all"]["jaccard"]["n"], 3)
        self.assertAlmostEqual(
            summary["correlations"]["all"]["jaccard"]["spearman_rho"], 1.0
        )
        self.assertEqual(summary["correlations"]["all"]["hss"]["n"], 2)
        self.assertAlmostEqual(
            summary["correlations"]["all"]["hss"]["spearman_rho"], 1.0
        )

    def test_rejects_duplicate_or_incomplete_pair_matrix(self):
        row_ab = {
            "model_a": "v/a",
            "model_b": "v/b",
            "jsd": 0.1,
            "jaccard": 0.9,
            "hss": 0.8,
            "lift": 2.0,
            "both_wrong": 20,
        }
        row_ac = {
            **row_ab,
            "model_b": "v/c",
        }

        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_alignment([row_ab, row_ab])
        with self.assertRaisesRegex(ValueError, "complete matrix"):
            summarize_alignment([row_ab, row_ac])

    def test_renders_publication_table_with_pair_counts(self):
        summary = {
            "correlations": {
                "all": {
                    "jaccard": {"n": 100, "spearman_rho": 0.294},
                    "hss": {"n": 80, "spearman_rho": 0.107},
                },
                "same_vendor": {
                    "jaccard": {"n": 20, "spearman_rho": 0.334},
                    "hss": {"n": 15, "spearman_rho": 0.256},
                },
                "cross_vendor": {
                    "jaccard": {"n": 80, "spearman_rho": 0.272},
                    "hss": {"n": 65, "spearman_rho": 0.070},
                },
            }
        }

        table = render_latex_table(summary)

        self.assertIn("All pairs & Jaccard & 100 & 0.294", table)
        self.assertIn("Same vendor & HSS & 15 & 0.256", table)
        self.assertIn("Cross vendor & HSS & 65 & 0.070", table)
        self.assertIn("label{tab:single-token-complementarity}", table)

    def test_verifies_matrix_digest_before_parsing(self):
        raw = (
            b'model,"vendor/a","vendor/b"\n'
            b'"vendor/a",0,0.1\n'
            b'"vendor/b",0.1,0\n'
        )
        digest = hashlib.sha256(raw).hexdigest()

        distances, actual_digest = read_verified_distance_matrix(
            raw,
            expected_sha256=digest,
        )

        self.assertEqual(distances, {("vendor/a", "vendor/b"): 0.1})
        self.assertEqual(actual_digest, digest)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            read_verified_distance_matrix(raw)
        with self.assertRaisesRegex(ValueError, "64-character hex digest"):
            read_verified_distance_matrix(raw, expected_sha256="not-a-digest")

    def test_committed_result_matches_pinned_integrity_regression(self):
        result_path = PROJECT_ROOT / "results" / "single_token_complementarity.json"
        result = json.loads(result_path.read_text())
        expected = {
            "n_models": 82,
            "n_pairs": 3321,
            "min_joint_wrong_for_hss": 10,
            "correlations": {
                "all": {
                    "jaccard": {"n": 3321, "spearman_rho": 0.282},
                    "hss": {"n": 2598, "spearman_rho": 0.074},
                },
                "same_vendor": {
                    "jaccard": {"n": 276, "spearman_rho": 0.374},
                    "hss": {"n": 219, "spearman_rho": 0.261},
                },
                "cross_vendor": {
                    "jaccard": {"n": 3045, "spearman_rho": 0.256},
                    "hss": {"n": 2379, "spearman_rho": 0.039},
                },
            },
            "source": {
                "single_token_dataset_doi": "10.5281/zenodo.21278557",
                "single_token_matrix_member": "results/divergence-matrix.csv",
                "single_token_matrix_sha256": PINNED_SINGLE_TOKEN_MATRIX_SHA256,
                "ikp_fingerprint_results": (
                    "results/comprehensive_fingerprint_results.json"
                ),
                "ikp_fingerprint_results_sha256": (
                    "2837a6f9bdc35b79adfc805bbe5ba4d342a732eb351fd3bd54049d23d433372d"
                ),
                "alignment": (
                    "exact served model identifier; non-thinking IKP runs only"
                ),
            },
        }

        self.assertEqual(result, expected)
        self.assertRegex(
            result["source"]["single_token_matrix_sha256"],
            r"\A[0-9a-f]{64}\Z",
        )
        self.assertEqual(
            result["n_pairs"],
            result["n_models"] * (result["n_models"] - 1) // 2,
        )
        knowledge_path = PROJECT_ROOT / result["source"]["ikp_fingerprint_results"]
        self.assertEqual(
            hashlib.sha256(knowledge_path.read_bytes()).hexdigest(),
            result["source"]["ikp_fingerprint_results_sha256"],
        )
        table_path = (
            PROJECT_ROOT / "results" / "tables" / "single_token_complementarity.tex"
        )
        self.assertEqual(table_path.read_text(), render_latex_table(result))


if __name__ == "__main__":
    unittest.main()
