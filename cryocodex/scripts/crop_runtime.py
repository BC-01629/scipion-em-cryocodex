"""Prepare a crop preview or run the unchanged backend with confirmed bounds.

Invoked with CryoCodex's Python, not Scipion's. No GUI or Scipion imports here.
The adapter is scoped to this subprocess; installed backend files are untouched.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

if __package__:
    from .crop_support import SCHEMA, backend_signature, read_record, map_signature, write_json, validate_bounds
else:
    from crop_support import SCHEMA, backend_signature, read_record, map_signature, write_json, validate_bounds


def prepare(args):
    import numpy as np
    from data_processor import DateProcessor
    from crop_check import mask_bounds, make_projections

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    grid, mask, origin, voxel = DateProcessor().gen_mask(args.input)
    if not np.isfinite(grid).all() or not np.any(grid > 0):
        raise ValueError('Crop preview requires finite, positive density.')
    np.savez_compressed(out / 'projections.npz', **make_projections(grid))
    record = dict(schema=SCHEMA, shape=[int(n) for n in grid.shape],
                  bounds={a: list(p) for a, p in mask_bounds(mask).items()},
                  origin=[float(x) for x in origin], voxel_size=[float(x) for x in voxel],
                  input_sha256=map_signature(args.input), backend=backend_signature(args.backend_home))
    write_json(out / 'preview.json', record)
    print('Crop preview ready.', flush=True)


def install_review(predict, record):
    """Reuse backend inference, substituting only the reviewed crop selection."""
    import numpy as np

    base = type(predict.data_processor)

    class ReviewedProcessor(base):
        def gen_mask(self, path):
            result = super().gen_mask(path)
            grid, mask, origin, voxel = result
            if (list(grid.shape) != record['shape']
                    or not np.allclose(origin, record['origin'], atol=1e-5)
                    or not np.allclose(voxel, record['voxel_size'], atol=1e-5)):
                raise ValueError('Prepared grid changed since crop confirmation. Review it again.')
            validate_bounds(record['bounds'], list(grid.shape))
            return result

        def crop_by_mask(self, grid, mask_ft, pad=5, bounds=None):
            print('# Using the crop confirmed in Scipion: ' + json.dumps(record['bounds']), flush=True)
            return super().crop_by_mask(grid, mask_ft, pad=pad, bounds=record['bounds'])

    predict.data_processor = ReviewedProcessor(target_voxel_size=1.0)
    reconstruct = predict.grid_reconstruction_gaussian_multi
    shape = tuple(record['bounds'][a][1] - record['bounds'][a][0] + 1 for a in 'zyx')

    def reconstruct_without_padding(*args, **kwargs):
        # The backend pads crops smaller than its 48-voxel inference window.
        # Remove that computational padding before restoring the original canvas
        # or interpolating, so a small reviewed crop does not change its extent.
        arrays = reconstruct(*args, **kwargs)
        return tuple(a[:shape[0], :shape[1], :shape[2]] for a in arrays)

    predict.grid_reconstruction_gaussian_multi = reconstruct_without_padding


def run(args, backend_args):
    record = read_record(Path(args.review_file).read_text())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--in_map_path', required=True)
    parser.add_argument('--normal', default='False')
    parser.add_argument('--crop_check', default='False')
    options, _ = parser.parse_known_args(backend_args)
    if options.normal.lower() != 'false' or options.crop_check.lower() != 'false':
        raise ValueError('Reviewed inference requires Fast mode and no terminal crop prompt.')
    if map_signature(options.in_map_path) != record['input_sha256']:
        raise ValueError('Input changed since crop confirmation. Review the crop again.')
    if backend_signature(args.backend_home) != record['backend']:
        raise ValueError('CryoCodex preprocessing changed. Review the crop again.')
    predict = importlib.import_module('predict')
    install_review(predict, record)
    sys.argv = [str(Path(args.backend_home) / 'predict.py')] + backend_args
    predict.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--backend-home', required=True)
    prep.add_argument('--input', required=True)
    prep.add_argument('--output', required=True)
    infer = commands.add_parser('run')
    infer.add_argument('--backend-home', required=True)
    infer.add_argument('--review-file', required=True)
    args, remaining = parser.parse_known_args()
    args.backend_home = os.path.abspath(args.backend_home)
    sys.path.insert(0, args.backend_home)
    if args.command == 'prepare':
        if remaining:
            parser.error('Unexpected preview arguments: ' + ' '.join(remaining))
        prepare(args)
    else:
        run(args, remaining[1:] if remaining[:1] == ['--'] else remaining)


if __name__ == '__main__':
    main()
