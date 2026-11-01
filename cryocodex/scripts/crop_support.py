"""Crop records shared by Scipion (Python 3.8) and the inference interpreter."""

import hashlib
import json
from pathlib import Path

SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def map_signature(path):
    """Hash scientific input, excluding MRC creation timestamps and labels."""
    import mrcfile
    import numpy as np
    digest = hashlib.sha256()
    with mrcfile.mmap(path, mode='r') as mrc:
        geometry = dict(shape=list(mrc.data.shape),
                        voxel=[float(mrc.voxel_size[a]) for a in 'xyz'],
                        origin=[float(mrc.header.origin[a]) for a in 'xyz'],
                        axes=[int(mrc.header[a]) for a in ('mapc', 'mapr', 'maps')],
                        starts=[int(mrc.header[a]) for a in ('nxstart', 'nystart', 'nzstart')])
        digest.update(json.dumps(geometry, sort_keys=True).encode('ascii'))
        for plane in mrc.data:
            digest.update(np.asarray(plane, dtype='<f4', order='C').tobytes())
    return digest.hexdigest()


def backend_signature(home):
    return {name: sha256_file(Path(home) / name)
            for name in ('predict.py', 'data_processor.py', 'grid_utils.py', 'crop_check.py')}


def validate_bounds(bounds, shape):
    if len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError('Invalid crop grid dimensions.')
    if not isinstance(bounds, dict) or set(bounds) != set('xyz'):
        raise ValueError('The crop must contain X, Y and Z ranges.')
    result = {}
    for axis, size in zip('xyz', reversed(shape)):
        pair = bounds[axis]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(v) is not int for v in pair)
                or not 0 <= pair[0] <= pair[1] < size):
            raise ValueError('%s bounds must satisfy 0 <= min <= max < %d.' % (axis.upper(), size))
        result[axis] = list(pair)
    return result


def read_record(value):
    record = json.loads(value) if isinstance(value, str) else value
    if not isinstance(record, dict) or record.get('schema') != SCHEMA:
        raise ValueError('Missing or unsupported crop confirmation. Review the crop again.')
    validate_bounds(record['bounds'], record['shape'])
    if not record.get('input_sha256') or not record.get('backend'):
        raise ValueError('The crop confirmation has no input/backend identity.')
    return record


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)
