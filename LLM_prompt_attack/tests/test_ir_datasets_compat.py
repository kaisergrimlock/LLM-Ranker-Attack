"""Regression tests for native-Windows ir_datasets download handling."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_datasets_compat import enable_windows_download_compat  # noqa: E402


@unittest.skipUnless(os.name == "nt", "Windows file locking regression")
class WindowsDownloadCompatibilityTests(unittest.TestCase):
    def test_ir_datasets_proxy_returns_a_closed_replaceable_file(self):
        """The placeholder must be closed before ir_datasets calls os.replace."""
        fake_download_module = SimpleNamespace(tempfile=tempfile)
        self.assertTrue(enable_windows_download_compat(fake_download_module))

        with tempfile.TemporaryDirectory() as directory:
            handle = fake_download_module.tempfile.NamedTemporaryFile(
                delete=False, dir=directory
            )
            source = f"{handle.name}.replacement"
            Path(source).write_bytes(b"complete")

            self.assertTrue(handle.closed)
            os.replace(source, handle.name)
            self.assertEqual(Path(handle.name).read_bytes(), b"complete")

    def test_patch_is_idempotent(self):
        """Repeated imports must not wrap the same module more than once."""
        fake_download_module = SimpleNamespace(tempfile=tempfile)
        self.assertTrue(enable_windows_download_compat(fake_download_module))
        self.assertFalse(enable_windows_download_compat(fake_download_module))
