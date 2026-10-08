"""Standard-library-only checks; never import or run the scientific pipeline."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE = PACKAGE_ROOT / "code"
spec = importlib.util.spec_from_file_location("timc_paths_under_test", CODE / "timc_paths.py")
paths = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(paths)
manifest_spec = importlib.util.spec_from_file_location("package_manifest_under_test", CODE / "package_manifest.py")
manifest_files = importlib.util.module_from_spec(manifest_spec)
assert manifest_spec.loader is not None
manifest_spec.loader.exec_module(manifest_files)


class CodePathTests(unittest.TestCase):
    def setUp(self):
        # Keep temporary synthetic files inside the test directory.
        self.temp = tempfile.TemporaryDirectory(prefix="code_paths_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.test_root = Path(self.temp.name).resolve()
        self.package = self.test_root / "package"
        self.package.mkdir()
        self.caller = self.test_root / "caller"
        self.caller.mkdir()
        old_cwd = Path.cwd()
        os.chdir(self.caller)
        self.addCleanup(os.chdir, old_cwd)
        self.package_patch = patch.object(paths, "PACKAGE_ROOT", self.package)
        self.profile_patch = patch.object(paths, "DEFAULT_PROFILE", self.package / "migration_paths.json")
        self.package_patch.start()
        self.profile_patch.start()
        self.addCleanup(self.package_patch.stop)
        self.addCleanup(self.profile_patch.stop)

    def write_profile(self, external, filename="migration_paths.json"):
        path = self.package / filename
        path.write_text(json.dumps({"external": external}), encoding="utf-8")
        return path

    def test_package_resource_ignores_caller_directory(self):
        self.assertEqual(paths.package_path("models/final_thickness_model.pt"),
                         self.package / "models" / "final_thickness_model.pt")

    def test_external_relative_path_uses_caller_directory(self):
        self.assertEqual(paths.external_path("private/archive"), self.caller / "private" / "archive")

    def test_absolute_overrides_are_preserved(self):
        target = self.test_root / "private" / "model.pt"
        self.assertEqual(paths.external_path(target), target)
        self.assertEqual(paths.package_path(target), target)

    def test_profile_values_use_profile_directory(self):
        self.write_profile({"archive_root": "private/archive", "data1_root": "private/source"})
        self.assertEqual(paths.archive_directory(environ={}), self.package / "private" / "archive")
        self.assertEqual(paths.data_directory("Data1", environ={}), self.package / "private" / "source")

    def test_explicit_environment_overrides_profile(self):
        self.write_profile({"data1_root": "profile_source"})
        self.assertEqual(paths.data_directory("Data1", environ={"ROLLING_DATA1_DIR": "env_source"}),
                         self.caller / "env_source")

    def test_profile_override_is_caller_relative(self):
        path = self.write_profile({"data2_root": "data"}, "override.json")
        self.assertEqual(paths.data_directory("Data2", environ={"TIMC_PATHS_CONFIG": "../package/override.json"}),
                         path.parent / "data")

    def test_missing_optional_profile_uses_package_defaults(self):
        self.assertEqual(paths.data_directory("Data1", environ={}), self.package / "data1")
        self.assertEqual(paths.data_directory("Data2", environ={}), self.package / "data2")
        self.assertIsNone(paths.archive_directory(environ={}))

    def test_missing_explicit_profile_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            paths.paths_profile({"TIMC_PATHS_CONFIG": "missing.json"})

    def test_required_data_path_remains_required(self):
        with self.assertRaises(ValueError):
            paths.data_directory("Data1", required=True, environ={})
        with self.assertRaises(ValueError):
            paths.archive_directory(required=True, environ={})

    def test_null_profile_fields_do_not_become_literal_paths(self):
        self.write_profile({"archive_root": None, "data1_root": ""})
        self.assertIsNone(paths.archive_directory(environ={}))
        self.assertEqual(paths.data_directory("Data1", environ={}), self.package / "data1")

    def test_invalid_profile_shape_and_path_type_are_reported(self):
        profile = self.package / "migration_paths.json"
        profile.write_text('{"external": []}', encoding="utf-8")
        with self.assertRaises(ValueError):
            paths.paths_profile({})
        self.write_profile({"data1_root": 123})
        with self.assertRaises(ValueError):
            paths.data_directory("Data1", environ={})

    def test_scale_lock_default_and_external_override(self):
        self.assertEqual(paths.scale_lock_path({}), self.package / "config" / "data1_scale_lock.json")
        self.assertEqual(paths.scale_lock_path({"TIMC_DATA1_SCALE_LOCK": "custom/lock.json"}),
                         self.caller / "custom" / "lock.json")

    def test_child_environment_preserves_caller_relative_paths(self):
        original = {"ROLLING_DATA1_DIR": "data1", "ROLLING_DATA2_DIR": "data2",
                    "TIMC_DATA1_SCALE_LOCK": "lock.json", "TIMC_PATHS_CONFIG": "profile.json",
                    "OTHER_SETTING": "retained"}
        child = paths.normalized_path_environment(original)
        self.assertEqual(child["ROLLING_DATA1_DIR"], str(self.caller / "data1"))
        self.assertEqual(child["ROLLING_DATA2_DIR"], str(self.caller / "data2"))
        self.assertEqual(child["TIMC_DATA1_SCALE_LOCK"], str(self.caller / "lock.json"))
        self.assertEqual(child["TIMC_PATHS_CONFIG"], str(self.caller / "profile.json"))
        self.assertEqual(child["OTHER_SETTING"], "retained")
        self.assertEqual(original["ROLLING_DATA1_DIR"], "data1")

    def test_profile_lookup_survives_child_working_directory_change(self):
        self.write_profile({"data2_root": "private/data2"}, "override.json")
        env = {"TIMC_PATHS_CONFIG": "../package/override.json", "ROLLING_DATA1_DIR": "private/data1"}
        before = (paths.data_directory("Data1", environ=env),
                  paths.data_directory("Data2", environ=env))
        child = paths.normalized_path_environment(env)
        os.chdir(self.package)
        after = (paths.data_directory("Data1", environ=child),
                 paths.data_directory("Data2", environ=child))
        self.assertEqual(after, before)

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            paths.data_directory("Data3", environ={})

    def test_changed_scientific_modules_parse_without_importing_dependencies(self):
        for name in ["timc_paths.py", "controller_replay.py", "protocol.py", "model_lopo.py",
                     "model_lopo_data.py", "stress_sweep.py", "batch_archive.py", "run_pipeline.py",
                     "engine_core/20_independent_replay_validation.py", "package_manifest.py", "publish_tables.py"]:
            with self.subTest(name=name):
                ast.parse((CODE / name).read_text(encoding="utf-8-sig"), filename=name)
        ast.parse((PACKAGE_ROOT / "verify_results.py").read_text(encoding="utf-8-sig"))


class StandaloneRepositoryTests(unittest.TestCase):
    def test_default_profile_is_inside_repository_root(self):
        self.assertEqual(paths.PACKAGE_ROOT, PACKAGE_ROOT)
        self.assertEqual(paths.DEFAULT_PROFILE, PACKAGE_ROOT / "migration_paths.json")

    def test_public_example_has_no_machine_paths(self):
        payload = json.loads((PACKAGE_ROOT / "migration_paths.example.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["external"], {
            "archive_root": None, "data1_root": None, "data2_root": None,
        })

    def test_manifest_includes_support_files_and_excludes_private_inputs(self):
        with tempfile.TemporaryDirectory(prefix="manifest_paths_", dir=Path(__file__).parent) as temp:
            root = Path(temp)
            included = [".gitattributes", ".gitignore", "SUBMISSION_VERSION.md",
                        "migration_paths.example.json", "code/timc_paths.py", "models/final.pt"]
            excluded = ["MANIFEST_SHA256.csv", "migration_paths.json", "migration_paths.local-dev.json",
                        "outputs/trace.csv", "data1/input.csv", "data2/input.csv",
                        "private_data/input.csv", "raw_data/input.csv", "batch_archive/input.csv",
                        "production_batch_root/input.csv", ".venv/config.json", "venv/config.json",
                        ".git/config", "__pycache__/cached.pyc", ".pytest_cache/cache.json",
                        "trace.log", "trace.csv.gz"]
            for relative in included + excluded:
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("synthetic test content", encoding="utf-8")
            found = {file.relative_to(root).as_posix() for file in manifest_files.package_files(root)}
            self.assertEqual(found, set(included))

    def test_manifest_writer_and_verifier_share_file_enumerator(self):
        for relative in ["code/publish_tables.py", "verify_results.py"]:
            with self.subTest(relative=relative):
                tree = ast.parse((PACKAGE_ROOT / relative).read_text(encoding="utf-8-sig"))
                self.assertTrue(any(
                    isinstance(node, ast.ImportFrom) and node.module == "package_manifest"
                    and any(alias.name == "package_files" for alias in node.names)
                    for node in ast.walk(tree)
                ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
