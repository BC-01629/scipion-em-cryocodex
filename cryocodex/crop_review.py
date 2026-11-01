"""Map identity and crop confirmation, independent of the desktop interface."""

import json
import os
from pathlib import Path

from . import Plugin
from .scripts.crop_support import backend_signature, read_record


def runtime_script():
    return str(Path(__file__).parent / 'scripts' / 'crop_runtime.py')


def input_identity(protocol):
    volume = protocol.inputVolume.get()
    if volume is None:
        raise ValueError('Select an imported input volume before checking the crop.')
    path = volume.getFileName()
    if path.endswith(':mrc'):
        path = path[:-4]
    if not os.path.isabs(path):
        path = os.path.join(protocol.getProject().path, path)
    path = os.path.realpath(path)
    stat = os.stat(path)
    # origin = volume.getOrigin()  # Restore with origin selection.
    return dict(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                index=volume.getIndex(), sampling=volume.getSamplingRate(),
                # origin_source=protocol.originSource.get(), coordinate_version=2,
                # origin=None if origin is None else [float(v) for v in origin.getShifts()])
                coordinate_version=3)


def confirmed_record(protocol):
    record = read_record(protocol.cropReview.get() or '')
    if record.get('input_identity') != input_identity(protocol):
        raise ValueError('Input map or its metadata changed. Check the crop again.')
    if record['backend'] != backend_signature(Plugin.getHome()):
        raise ValueError('CryoCodex preprocessing changed. Check the crop again.')
    return record


def record_is_current(protocol):
    try:
        confirmed_record(protocol)
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False
