"""Wizard discovery must work before and after the project GUI is imported."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestWizardDiscovery(unittest.TestCase):
    def test_import_orders(self):
        root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
        for first in ('', 'import pyworkflow.gui.project\n'):
            with self.subTest(project_first=bool(first)):
                result = subprocess.run(
                    [sys.executable, '-c', first + '''
import cryocodex.wizards as w
from pyworkflow.gui.form import FormWindow
callback = FormWindow._checkChanges
assert callback._cryocodex_auto_open
w._install_auto_open()
assert FormWindow._checkChanges is callback
'''], env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
