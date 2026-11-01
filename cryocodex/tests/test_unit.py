import contextlib
import io
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock

import mrcfile
import numpy as np
from pwem.objects import Volume, VolumeMask, Transform

from cryocodex import Plugin
from cryocodex.constants import OUTPUT_FILES, SOURCE_COMMIT, SOURCE_URL, SOURCE_BUILD_DIR, SOURCE_TAG
from cryocodex.convert import prepare_input, read_geometry
from cryocodex.protocols import ProtCryoCodexEnhance
from cryocodex.scripts import install_runtime


def write_map(path, data=None, sampling=1.5, origin=(1, 2, 3), starts=(0, 0, 0), axes=(1, 2, 3)):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.ones((8, 10, 12), dtype=np.float32) if data is None else data)
        mrc.voxel_size = sampling
        mrc.header.origin = origin
        mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = starts
        mrc.header.mapc, mrc.header.mapr, mrc.header.maps = axes


class TestMapConversion(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_axis_order_start_indices_and_origin(self):
        path = self.root / "source.mrc"
        write_map(path, origin=(0, 0, 0), starts=(10, 20, 30), axes=(2, 1, 3))
        geometry = read_geometry(str(path))
        self.assertEqual(geometry["dimensions"], (10, 12, 8))
        np.testing.assert_allclose(geometry["origin"], (30, 15, 45))

    def test_explicit_origin_is_not_added_to_start_indices(self):
        path = self.root / 'both.mrc'
        write_map(path, origin=(4, -2, 3), starts=(10, 20, 30), axes=(2, 1, 3))
        np.testing.assert_allclose(read_geometry(str(path))['origin'], (4, -2, 3))

    def test_default_preserves_file_origin_despite_scipion_centering(self):
        path = self.root / 'source.mrc'
        dest = self.root / 'input.mrc'
        volume = Volume()
        volume.setFileName(str(path))
        volume.setSamplingRate(1.5)
        origin = Transform()
        origin.setShifts(-9, -7.5, -6)
        volume.setOrigin(origin)
        for starts, expected in [((0, 0, 0), (0, 0, 0)), ((3, -2, 5), (4.5, -3, 7.5))]:
            write_map(path, origin=(0, 0, 0), starts=starts)
            prepare_input(volume, str(dest))
            np.testing.assert_allclose(read_geometry(str(dest))['origin'], expected)
            with mrcfile.open(str(dest)) as m:
                self.assertEqual((int(m.header.nxstart), int(m.header.nystart), int(m.header.nzstart)), starts)

    def test_input_copy_preserves_header_fields_without_modifying_source(self):
        path = self.root / "source.mrc"
        dest = self.root / "input.mrc"
        data = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
        write_map(path, data, axes=(2, 1, 3), starts=(10, 20, 30))
        before = path.read_bytes()
        volume = Volume()
        volume.setFileName(str(path))
        volume.setSamplingRate(2.25)
        origin = Transform()
        origin.setShifts(4.5, -9.0, 2.25)
        volume.setOrigin(origin)
        prepare_input(volume, str(dest))
        self.assertEqual(path.read_bytes(), before)
        info = read_geometry(str(dest))
        self.assertAlmostEqual(info["sampling"], 2.25)
        np.testing.assert_allclose(info["origin"], (1, 2, 3))
        with mrcfile.open(str(dest)) as mrc:
            np.testing.assert_array_equal(mrc.data, np.transpose(data, (0, 2, 1)))
            self.assertEqual((int(mrc.header.nxstart), int(mrc.header.nystart), int(mrc.header.nzstart)), (20, 10, 30))

    def test_anisotropic_map_rejected(self):
        path = self.root / "anisotropic.mrc"
        write_map(path)
        with mrcfile.open(str(path), mode="r+") as mrc:
            mrc.voxel_size = (1, 2, 3)
        with self.assertRaisesRegex(ValueError, "isotropic"):
            read_geometry(str(path))

    def test_zero_density_rejected(self):
        path = self.root / "empty.mrc"
        write_map(path, np.zeros((8, 8, 8), dtype=np.float32))
        volume = Volume()
        volume.setFileName(str(path))
        volume.setSamplingRate(1.5)
        with self.assertRaisesRegex(ValueError, "positive density"):
            prepare_input(volume, str(self.root / "copy.mrc"))


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cryocodex 50% '")
        self.addCleanup(self.temp.cleanup)
        self.protocol = ProtCryoCodexEnhance(workingDir=self.temp.name)
        os.makedirs(self.protocol._getExtraPath(), exist_ok=True)

    def test_arguments_quote_paths_and_use_assigned_gpu(self):
        self.protocol.mode.set(1)
        self.protocol.keepSize.set(True)  # hidden stale UI value must not reach normal mode
        with mock.patch.object(Plugin, "getModel", return_value="/model 50% 'test.pth"):
            args = shlex.split(self.protocol._arguments() % {"GPU": "3"})
        self.assertEqual(args[args.index("-g") + 1], "3")
        self.assertEqual(args[args.index("-m") + 1], "/model 50% 'test.pth")
        self.assertEqual(args[args.index("--keep_size") + 1], "False")
        self.assertEqual(args[args.index("--crop_check") + 1], "False")

    def test_invalid_parameters(self):
        self.protocol.batchSize.set(0)
        self.protocol.stride.set(49)
        self.protocol.gpuList.set("0 1")
        errors = self.protocol._validate()
        self.assertEqual(len(errors), 3)

    def test_four_outputs_use_real_metadata_and_mask_type(self):
        for filename in OUTPUT_FILES.values():
            write_map(self.protocol._getExtraPath(filename), sampling=1.25, origin=(0, 0, 0), starts=(3, -2, 5))
        with mock.patch.object(self.protocol, "_defineOutputs") as register, \
                mock.patch.object(self.protocol, "_defineSourceRelation") as relate:
            self.protocol.createOutputStep()
        outputs = register.call_args.kwargs
        self.assertEqual(set(outputs), set(OUTPUT_FILES))
        self.assertIsInstance(outputs["molecular_mask"], VolumeMask)
        for output in outputs.values():
            self.assertAlmostEqual(output.getSamplingRate(), 1.25)
            self.assertIsNone(output.getOrigin())
        self.assertEqual(relate.call_count, 4)

    def test_missing_or_inconsistent_outputs_do_not_register(self):
        with mock.patch.object(self.protocol, "_defineOutputs") as register:
            with self.assertRaisesRegex(RuntimeError, "required output"):
                self.protocol.createOutputStep()
            register.assert_not_called()
        for filename in OUTPUT_FILES.values():
            write_map(self.protocol._getExtraPath(filename))
        write_map(self.protocol._getExtraPath("cryocodex_in_mask.mrc"), sampling=2)
        with mock.patch.object(self.protocol, "_defineOutputs") as register:
            with self.assertRaisesRegex(RuntimeError, "inconsistent grids"):
                self.protocol.createOutputStep()
            register.assert_not_called()


class TestInstallation(unittest.TestCase):
    def test_installer_is_pinned_and_separate_from_scipion(self):
        env = mock.Mock()
        with mock.patch.object(Plugin, "getVar", side_effect=lambda name: "/opt/conda/bin/conda" if name.endswith("CONDA_EXE") else ""):
            Plugin.defineBinaries(env)
        args, kwargs = env.addPackage.call_args
        self.assertEqual(args, ("cryocodex",))
        self.assertEqual(kwargs["url"], SOURCE_URL)
        self.assertEqual(kwargs["buildDir"], SOURCE_BUILD_DIR)
        self.assertEqual(SOURCE_BUILD_DIR, 'CryoCodex-1.0.0')
        self.assertTrue(SOURCE_URL.endswith('/archive/refs/tags/v1.0.0.tar.gz'))
        self.assertIn('--source-tag ' + SOURCE_TAG, kwargs['commands'][0][0])
        self.assertIn("install_runtime.py", kwargs["commands"][0][0])
        self.assertIn("--conda /opt/conda/bin/conda", kwargs["commands"][0][0])

    def test_failed_download_leaves_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "CryoCodex.pth"
            response = io.BytesIO(b"incomplete")
            response.headers = {"Content-Length": "100"}
            with mock.patch.object(install_runtime.urllib.request, "urlopen", side_effect=lambda *a, **k: contextlib.nullcontext(response)), \
                    mock.patch.object(install_runtime.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    install_runtime.download_model("https://example.invalid/model", target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("CryoCodex.pth.part").exists())

    def test_dependency_failure_does_not_mark_installed(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("predict.py", "requirements.txt", "VQUnet.py"):
                (root / name).write_text("placeholder")
            python = root / "conda-env" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("placeholder")
            with mock.patch.object(install_runtime.shutil, "which", return_value="/conda"), \
                    mock.patch.object(install_runtime.subprocess, "run", side_effect=[None, subprocess.CalledProcessError(1, "pip")]):
                with self.assertRaises(subprocess.CalledProcessError):
                    install_runtime.install(root, "/conda", "https://example.invalid/model", SOURCE_COMMIT)
            self.assertFalse((root / "cryocodex-installed.json").exists())

    def test_backend_environment_excludes_scipion_python_paths(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/scipion/bindings", "PYTHONHOME": "/scipion", "CUDA_VISIBLE_DEVICES": "2"}), \
                mock.patch.object(Plugin, "getPython", return_value="/backend/bin/python"):
            env = Plugin.getEnviron()
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")


if __name__ == "__main__":
    unittest.main()
