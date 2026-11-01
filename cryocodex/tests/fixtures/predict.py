"""Offline subprocess fixture implementing only the tested CryoCodex CLI contract."""
import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np

parser = argparse.ArgumentParser()
for option in ("-i", "-o", "-n", "-m", "-b", "-s", "-g", "--normal", "--keep_size",
               "--reverse_interpolation", "--crop_check", "--no_logo"):
    parser.add_argument(option, required=True)
args = parser.parse_args()
assert args.crop_check == "False"
assert "%" not in args.g
assert Path(args.m).is_file()
with mrcfile.open(args.i) as source:
    assert source.data.ndim == 3
    assert int(source.header.nxstart) == 0
out_dir = Path(args.o)
out_dir.mkdir(exist_ok=True)
(out_dir / "invocation.json").write_text(json.dumps(vars(args)))
for suffix in ("", "_in_score", "_out_score", "_in_mask"):
    with mrcfile.new(str(out_dir / (args.n + suffix + ".mrc")), overwrite=True) as mrc:
        mrc.set_data(np.ones((8, 9, 10), dtype=np.float32))
        mrc.voxel_size = 1.25
        mrc.header.origin = (0, 0, 0)
        mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = 3, -2, 5
