"""MRC coordinates are read from the actual output, including start indices."""

import os

import mrcfile
import numpy as np


def map_geometry(mrc):
    axes = np.array([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=int)
    if sorted(axes.tolist()) != [1, 2, 3]:
        raise ValueError("MRC axis order must be a permutation of 1, 2, 3.")
    order = np.argsort(axes)
    voxel = np.array([float(mrc.voxel_size[a]) for a in ("x", "y", "z")])
    if not np.all(np.isfinite(voxel)) or np.any(voxel <= 0):
        raise ValueError("MRC voxel size must be finite and positive.")
    if not np.allclose(voxel, voxel[0], atol=1e-5, rtol=1e-5):
        raise ValueError("Scipion Volume requires isotropic sampling; resample this map first.")
    angles = [float(mrc.header.cellb[a]) for a in ("alpha", "beta", "gamma")]
    if not np.allclose(angles, 90):
        raise ValueError("CryoCodex requires an orthogonal MRC grid.")
    starts = np.array([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart])[order]
    origin = np.array([float(mrc.header.origin[a]) for a in ("x", "y", "z")])
    # Match MRC2000 viewers (including ChimeraX): a nonzero ORIGIN takes
    # precedence; otherwise use axis-mapped start indices. Do not add both.
    if not np.any(origin):
        origin = starts * voxel
    if not np.all(np.isfinite(origin)):
        raise ValueError("MRC origin must be finite.")
    dims = np.array([mrc.header.nx, mrc.header.ny, mrc.header.nz], dtype=int)[order]
    if np.any(dims <= 0) or mrc.data is None or mrc.data.ndim != 3:
        raise ValueError("Expected a nonempty three-dimensional MRC volume.")
    return {"sampling": float(voxel[0]), "origin": tuple(origin),
            "dimensions": tuple(int(d) for d in dims), "order": order}


def read_geometry(path):
    with mrcfile.open(path, header_only=False) as mrc:
        return map_geometry(mrc)


# Original signature for re-enabling origin selection:
# def prepare_input(volume, destination, use_header_origin=True):
def prepare_input(volume, destination):
    """Copy into the workspace without plugin-level origin correction.

    Canonicalize axes, retaining raw origin and axis-mapped start indices.
    Never edit the original imported map, including when it is a symbolic link.
    """
    source = volume.getFileName()
    if source.endswith(":mrc"):
        source = source[:-4]
    converted = None
    if not source.lower().endswith((".mrc", ".map", ".ccp4")) or volume.getIndex() not in (0, None):
        from pwem.emlib.image import ImageHandler
        converted = destination + ".converted.mrc"
        ImageHandler().convert(volume, converted)
        source = converted
    try:
        with mrcfile.open(source) as mrc:
            geometry = map_geometry(mrc)
            raw_origin = tuple(float(mrc.header.origin[a]) for a in ("x", "y", "z"))
            starts = np.array([mrc.header.nxstart, mrc.header.nystart,
                               mrc.header.nzstart])[geometry["order"]]
            data = np.ascontiguousarray(np.transpose(mrc.data, 2 - geometry["order"][::-1]),
                                        dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError("Input volume contains NaN or infinite values.")
        if not np.any(data > 0):
            raise ValueError("CryoCodex requires positive density values in the input map.")
        sampling = float(volume.getSamplingRate())
        if not np.isfinite(sampling) or sampling <= 0:
            raise ValueError("Input sampling rate must be finite and positive.")
        # Disabled origin selection/correction; retain for future use.
        # origin = volume.getOrigin()
        # shifts = (origin.getShifts() if not use_header_origin and origin is not None
                  # else geometry["origin"])
        # if not np.all(np.isfinite(shifts)):
            # raise ValueError("Input origin must be finite.")
        with mrcfile.new(destination, overwrite=True) as out:
            out.set_data(data)
            out.voxel_size = sampling
            # out.header.origin = tuple(float(x) for x in shifts)
            # out.header.nxstart = out.header.nystart = out.header.nzstart = 0
            out.header.origin = raw_origin
            out.header.nxstart, out.header.nystart, out.header.nzstart = starts
            out.update_header_stats()
    finally:
        if converted is not None and os.path.isfile(converted):
            os.unlink(converted)
