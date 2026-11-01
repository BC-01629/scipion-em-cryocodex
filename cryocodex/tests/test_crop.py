import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import mrcfile
from pwem.objects import Volume, Transform

from cryocodex import Plugin
from cryocodex.crop_review import input_identity, confirmed_record
from cryocodex.protocols import ProtCryoCodexEnhance
from cryocodex.scripts.crop_support import validate_bounds, read_record, map_signature
from cryocodex.scripts.crop_runtime import install_review


class TestCrop(unittest.TestCase):
    def test_map_identity_ignores_creation_label_but_detects_density_and_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.mrc'
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(np.ones((4, 4, 4), dtype=np.float32))
                mrc.voxel_size = 1.
            before = map_signature(path)
            with mrcfile.open(path, mode='r+') as mrc:
                mrc.header.label[0] = b'Created at another time'
            self.assertEqual(before, map_signature(path))
            with mrcfile.open(path, mode='r+') as mrc:
                mrc.data[0, 0, 0] = 2.
            self.assertNotEqual(before, map_signature(path))
            before = map_signature(path)
            with mrcfile.open(path, mode='r+') as mrc:
                mrc.header.origin.x = 1.
            self.assertNotEqual(before, map_signature(path))

    def record(self):
        return dict(schema=1, shape=[9, 10, 11], bounds=dict(x=[2, 7], y=[1, 8], z=[0, 4]),
                    origin=[0, 0, 0], voxel_size=[1, 1, 1], input_sha256='example', backend={'test': 'test'})

    def test_bounds_reject_out_of_grid_and_noninteger(self):
        for pair in ([-1, 4], [0, 11], [8, 7], [0., 4], [False, 4]):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                validate_bounds(dict(x=pair, y=[0, 9], z=[0, 8]), [9, 10, 11])
        self.assertEqual(validate_bounds(dict(x=[0, 10], y=[0, 9], z=[0, 8]), [9, 10, 11])['x'], [0, 10])

    def test_missing_confirmation_cannot_execute(self):
        prot = ProtCryoCodexEnhance(mode=0, checkCrop=True)
        self.assertTrue(any('Confirm the crop' in e for e in prot._validate()))
        prot.mode.set(1)
        self.assertFalse(prot._usesReviewedCrop())
        self.assertFalse(any('Confirm the crop' in e for e in prot._validate()))

    def test_review_arguments_use_adapter_and_no_stdin_prompt(self):
        prot = ProtCryoCodexEnhance(mode=0, checkCrop=True)
        prot.setWorkingDir('/tmp/crop test 20%')
        args = prot._arguments()
        self.assertIn('--review-file', args)
        self.assertIn('--crop_check False', args)
        self.assertIn('20%%', args)
        self.assertIn('%(GPU)s', args)

    def test_input_change_invalidates_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'volume.mrc'
            path.write_bytes(b'volume')
            volume = Volume()
            volume.setFileName(str(path))
            volume.setSamplingRate(1.5)
            origin = Transform()
            origin.setShifts(0, 0, 0)
            volume.setOrigin(origin)
            prot = ProtCryoCodexEnhance(inputVolume=volume, checkCrop=True)
            record = self.record()
            record['input_identity'] = input_identity(prot)
            prot.cropReview.set(json.dumps(record))
            with mock.patch('cryocodex.crop_review.backend_signature', return_value=record['backend']):
                self.assertEqual(confirmed_record(prot)['bounds'], record['bounds'])
                volume.setSamplingRate(2.)
                with self.assertRaisesRegex(ValueError, 'metadata changed'):
                    confirmed_record(prot)

    def test_adapter_uses_reviewed_bounds_and_removes_computational_padding(self):
        record = self.record()
        calls = []

        class Processor:
            def __init__(self, target_voxel_size=1.):
                pass

            def gen_mask(self, path):
                return np.ones(record['shape']), np.ones(record['shape']), [0, 0, 0], [1, 1, 1]

            def crop_by_mask(self, grid, mask, pad=5, bounds=None):
                calls.append(bounds)
                return grid, mask, np.array([2, 1, 0])

        backend = SimpleNamespace(data_processor=Processor(),
                                  grid_reconstruction_gaussian_multi=lambda: [np.ones((48, 48, 48))]*4)
        install_review(backend, record)
        grid, mask, _, _ = backend.data_processor.gen_mask('unused')
        backend.data_processor.crop_by_mask(grid, mask)
        self.assertEqual(calls, [record['bounds']])
        self.assertEqual([a.shape for a in backend.grid_reconstruction_gaussian_multi()], [(5, 8, 6)]*4)

    def test_changed_preprocessing_grid_is_rejected(self):
        class Processor:
            def __init__(self, target_voxel_size=1.):
                pass

            def gen_mask(self, path):
                return np.zeros((12, 10, 11)), None, [0, 0, 0], [1, 1, 1]

        backend = SimpleNamespace(data_processor=Processor(), grid_reconstruction_gaussian_multi=lambda: ())
        install_review(backend, self.record())
        with self.assertRaisesRegex(ValueError, 'Prepared grid changed'):
            backend.data_processor.gen_mask('unused')
