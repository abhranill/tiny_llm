"""Unit tests for TinyGPT checkpoint integrity validation."""

import tempfile
import unittest
from pathlib import Path

import validate_checkpoint


class CheckpointIntegrityTests(unittest.TestCase):
    def test_missing_files_are_reported_as_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results = validate_checkpoint.validate(
                root / "missing.pt",
                root / "missing_tokenizer.json",
                root / "missing_corpus.txt",
            )

        failures = [result for result in results if result[0] == "FAIL"]
        self.assertEqual(len(failures), 3)
        self.assertTrue(
            all("file exists" in result[1] for result in failures)
        )

    def test_baseline_checkpoint_passes_structural_checks(self):
        results = validate_checkpoint.validate(
            validate_checkpoint.DEFAULT_CHECKPOINT,
            validate_checkpoint.DEFAULT_TOKENIZER,
            validate_checkpoint.DEFAULT_CORPUS,
        )

        failed_names = [name for status, name, _ in results if status == "FAIL"]
        self.assertNotIn("Safe checkpoint loading", failed_names)
        self.assertNotIn("Required checkpoint keys", failed_names)
        self.assertNotIn(
            "Model reconstruction and state compatibility",
            failed_names,
        )

    def test_experiment_checkpoint_passes_corpus_hash_checks(self):
        checkpoint = Path("experiments/reproducible_training/best_model.pt")
        results = validate_checkpoint.validate(
            checkpoint,
            validate_checkpoint.DEFAULT_TOKENIZER,
            validate_checkpoint.DEFAULT_CORPUS,
        )

        checks = {
            name: status
            for status, name, _ in results
        }
        self.assertEqual(checks.get("Checkpoint corpus SHA-256"), "PASS")
        self.assertEqual(
            checks.get("Training metadata corpus SHA-256"),
            "PASS",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
