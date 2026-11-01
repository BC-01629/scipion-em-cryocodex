import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import uuid
from unittest import mock

import mrcfile
import numpy as np
from pyworkflow.tests import BaseTest, setupTestProject
from pwem.objects import Volume, VolumeMask, Transform
from pwem.protocols import ProtImportVolumes

from cryocodex import Plugin
from cryocodex.constants import OUTPUT_FILES
from cryocodex.convert import read_geometry
from cryocodex.protocols import ProtCryoCodexEnhance


def synthetic_input(path):
    z, y, x = np.indices((48, 48, 48), dtype=np.float32)
    data = np.exp(-((x - 24)**2 + (y - 24)**2 + (z - 24)**2) / 60).astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = 1.0


class TestCryoCodexIntegration(BaseTest):
    """Real Scipion execution and persistence, with a synthetic backend (no CUDA)."""
    @classmethod
    def setUpClass(cls):
        cls.__name__ = "TestCryoCodexIntegration_" + uuid.uuid4().hex[:8]
        setupTestProject(cls)
        cls.backend = Path(cls.proj.path) / "fixture-backend"
        cls.backend.mkdir()
        shutil.copyfile(Path(__file__).parent / "fixtures" / "predict.py", cls.backend / "predict.py")
        cls.model = cls.backend / "fixture.pth"
        cls.model.write_text("Synthetic test fixture; not a CryoCodex checkpoint.")
        cls.overrides = {"CRYOCODEX_HOME": str(cls.backend), "CRYOCODEX_PYTHON": sys.executable,
                         "CRYOCODEX_MODEL": str(cls.model)}
        cls.env_patch = mock.patch.dict(os.environ, cls.overrides)
        cls.env_patch.start()
        cls.addClassCleanup(cls.env_patch.stop)
        cls.vars_patch = mock.patch.dict(Plugin._vars, cls.overrides)
        cls.vars_patch.start()
        cls.addClassCleanup(cls.vars_patch.stop)
        source = str(Path(cls.proj.path) / "input.mrc")
        synthetic_input(source)
        imp = cls.newProtocol(ProtImportVolumes, filesPath=source, samplingRate=1.0)
        cls.launchProtocol(imp, wait=True)
        cls.inputVolume = imp.outputVolume

    def test_scipion_pipeline(self):
        protocol = self.newProtocol(ProtCryoCodexEnhance, inputVolume=self.inputVolume,
                                    batchSize=1, stride=24, mode=1)
        self.launchProtocol(protocol, wait=True)
        for name in OUTPUT_FILES:
            output = getattr(protocol, name)
            self.assertIsInstance(output, Volume)
            self.assertAlmostEqual(output.getSamplingRate(), 1.25)
            self.assertIsNone(output.getOrigin())
            self.assertTrue(os.path.isfile(output.getFileName()))
        self.assertIsInstance(protocol.molecular_mask, VolumeMask)
        invocation = json.loads(Path(protocol._getExtraPath("invocation.json")).read_text())
        self.assertEqual(invocation["normal"], "True")
        self.assertEqual(invocation["g"], "0")


@unittest.skipUnless(os.environ.get("CRYOCODEX_TEST_GPU") == "1",
                     "Set CRYOCODEX_TEST_GPU=1 in an allocated GPU session to run real inference.")
class TestCryoCodexGPU(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.__name__ = "TestCryoCodexGPU_" + uuid.uuid4().hex[:8]
        setupTestProject(cls)
        source = os.environ.get("CRYOCODEX_TEST_MAP")
        if source:
            source = os.path.abspath(os.path.expanduser(source))
            if not os.path.isfile(source):
                raise ValueError("CRYOCODEX_TEST_MAP must be an existing file.")
        else:
            source = str(Path(cls.proj.path) / "input.mrc")
            synthetic_input(source)
        sampling = float(os.environ.get("CRYOCODEX_TEST_APIX", read_geometry(source)["sampling"]))
        imp = cls.newProtocol(ProtImportVolumes, filesPath=source, samplingRate=sampling)
        cls.launchProtocol(imp, wait=True)
        cls.inputVolume = imp.outputVolume

    def test_real_inference(self):
        protocol = self.newProtocol(ProtCryoCodexEnhance, inputVolume=self.inputVolume,
                                    mode=1, batchSize=1, stride=24, reverseInterpolation=True,
                                    gpuList=os.environ.get("CRYOCODEX_TEST_GPU_ID", "0"))
        self.launchProtocol(protocol, wait=True)
        input_geometry = read_geometry(self.inputVolume.getFileName())
        for name in OUTPUT_FILES:
            output = getattr(protocol, name)
            geometry = read_geometry(output.getFileName())
            self.assertAlmostEqual(output.getSamplingRate(), geometry["sampling"])
            self.assertIsNone(output.getOrigin())
            self.assertAlmostEqual(geometry['sampling'], self.inputVolume.getSamplingRate(), places=5)
            # Regridding can round origins by a voxel; a half-box translation
            # must never be inherited from Scipion's automatic centering.
            np.testing.assert_allclose(geometry['origin'], input_geometry['origin'],
                                       atol=2 * max(1., input_geometry['sampling']))
            if np.allclose(input_geometry['origin'], 0):
                np.testing.assert_allclose(geometry['origin'], 0, atol=1e-5)
            with mrcfile.open(output.getFileName()) as mrc:
                self.assertTrue(np.isfinite(mrc.data).all())


@unittest.skipUnless(os.environ.get('CRYOCODEX_TEST_REVIEW'),
                     'Set CRYOCODEX_TEST_REVIEW to a GUI-confirmed crop record for a GPU test.')
class TestCryoCodexReviewedGPU(BaseTest):
    """Real GUI-selected crop, real inference, zero fill and native resampling."""

    @classmethod
    def setUpClass(cls):
        cls.__name__ = 'TestCryoCodexReviewedGPU_' + uuid.uuid4().hex[:8]
        cls.record = json.loads(Path(os.environ['CRYOCODEX_TEST_REVIEW']).read_text())
        setupTestProject(cls)
        identity = cls.record['input_identity']
        origin = (read_geometry(identity['path'])['origin'] if identity.get('origin_source', 0) == 0
                  else identity['origin'] or [0, 0, 0])
        imp = cls.newProtocol(ProtImportVolumes, filesPath=identity['path'],
                              samplingRate=identity['sampling'], setOrigCoord=True,
                              x=-origin[0], y=-origin[1], z=-origin[2])
        cls.launchProtocol(imp, wait=True)
        cls.inputVolume = imp.outputVolume

    def test_confirmed_crop_inference(self):
        from cryocodex.crop_review import input_identity
        protocol = self.newProtocol(ProtCryoCodexEnhance, inputVolume=self.inputVolume,
                                    mode=0, checkCrop=True, keepSize=True, reverseInterpolation=True,
                                    batchSize=1, stride=24,
                                    gpuList=os.environ.get('CRYOCODEX_TEST_GPU_ID', '0'))
        record = dict(self.record, input_identity=input_identity(protocol))
        protocol.cropReview.set(json.dumps(record))
        self.launchProtocol(protocol, wait=True)
        saved = json.loads(Path(protocol._getExtraPath('crop_review.json')).read_text())
        self.assertEqual(saved['bounds'], self.record['bounds'])
        for name in OUTPUT_FILES:
            volume = getattr(protocol, name)
            self.assertAlmostEqual(volume.getSamplingRate(), self.inputVolume.getSamplingRate(), places=5)
            with mrcfile.open(volume.getFileName()) as mrc:
                self.assertTrue(np.isfinite(mrc.data).all())
                if name == 'molecular_mask':
                    self.assertTrue(np.isin(mrc.data, [0, 1]).all())
