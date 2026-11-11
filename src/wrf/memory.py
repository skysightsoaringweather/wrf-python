"""Bounded-workspace variants of the existing CAPE and interpolation kernels."""

import operator
import threading

import numpy as np

from ._wrffortran import dcapemax3d, dinterp3dz_uvw
from .constants import Constants, default_fill
from .extension import DiagnosticError
from .util import psafilepath, npbytes_to_str

# The legacy pseudoadiabat reader uses a fixed Fortran unit. Serialize calls to
# this API; each native call still uses its configured OpenMP team.
_cape_lock = threading.Lock()


def _volumes(arrays, block_rows):
    block_rows = operator.index(block_rows)
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    arrays = [np.asarray(a) for a in arrays]
    if arrays[0].ndim != 3 or any(n == 0 for n in arrays[0].shape):
        raise ValueError("Expected nonempty (level, y, x) arrays")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("All fields must have the same (level, y, x) shape")
    if any(a.dtype not in (np.dtype('float32'), np.dtype('float64')) for a in arrays):
        raise TypeError("Fields must have float32 or float64 dtype")
    return arrays, block_rows


def cape_3d_max(pres_hpa, tkel, qv, height, *, block_rows=16):
    """Return max(cape_3d(..., meta=False)[0], axis=0) with bounded scratch.

    Accepts a single nonempty (level, y, x) grid, in either vertical order.
    As with cape_3d, inputs must contain no NaNs or missing values. The result
    has the pressure dtype and the default CAPE fill value/mask conventions.
    The legacy top-parcel zero is retained and missing parcel results are
    ignored. No full-domain CAPE/CIN volumes or float64 input copies are made.
    Terrain, surface pressure and terrain-following mode do not enter the
    all-parcel 3D calculation, so this API requires only its four dependencies.
    The original cape_3d and cape_2d APIs are unchanged.
    """
    fields = (pres_hpa, tkel, qv, height)
    if any(np.ma.is_masked(a) for a in fields):
        raise ValueError("CAPE inputs must contain no NaNs or missing values")
    arrays, block_rows = _volumes(fields, block_rows)
    p = arrays[0]
    reverse = p[0, 0, 0] > p[-1, 0, 0]
    vertical = slice(None, None, -1) if reverse else slice(None)
    output = np.empty(p.shape[1:], dtype=p.dtype)
    for y in range(0, p.shape[1], block_rows):
        inputs = [np.asfortranarray(a[vertical, y:y + block_rows, :].T, dtype=np.float64)
                  for a in arrays]
        if any(not np.isfinite(a).all() for a in inputs):
            raise ValueError("CAPE inputs must contain no NaNs or missing values")
        shape = inputs[0].shape[:2] + (1,)
        cape = np.empty(shape, dtype=np.float64, order='F')
        errstat = np.array(0, dtype=np.int32)
        errmsg = np.zeros(Constants.ERRLEN, 'c')
        with _cape_lock:
            dcapemax3d(*inputs, cape, default_fill(np.float64), psafilepath(), errstat, errmsg)
        if int(errstat):
            raise DiagnosticError(''.join(npbytes_to_str(errmsg)).strip())
        output[y:y + block_rows] = cape[:, :, 0].T
    return np.ma.masked_values(output, default_fill(np.float64), copy=False)


def interplevel_uvw(u, v, w, z, levels, *, out=None, block_rows=16):
    """Interpolate U/V/W using the same brackets and arithmetic as interplevel.

    Inputs are single (level, y, x) grids. Result buffers have shape
    (len(levels), y, x) and each wind component's dtype. Optional caller-owned
    buffers avoid copying complete profiles. Missing results are NaN, matching
    a filled interplevel result. Existing endpoint comparisons, the top-down
    search order (including inversions), and float64 weights are unchanged.
    """
    arrays, block_rows = _volumes((u, v, w, z), block_rows)
    u, v, w, z = arrays
    levels = np.asarray(levels, dtype=np.float64)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all():
        raise ValueError("Expected nonempty finite 1D levels")
    shape = (len(levels),) + z.shape[1:]
    if out is None:
        out = tuple(np.empty(shape, dtype=a.dtype) for a in arrays[:3])
    if len(out) != 3:
        raise ValueError("Expected three output buffers")
    for a, dest in zip(arrays[:3], out):
        if dest.shape != shape or dest.dtype != a.dtype or not dest.flags.writeable:
            raise ValueError("Output shape/dtype/writeability does not match its wind field")
        if any(np.may_share_memory(dest, source) for source in arrays):
            raise ValueError("Output buffers must not overlap inputs")
    if any(np.may_share_memory(out[i], out[j]) for i in range(3) for j in range(i)):
        raise ValueError("Output buffers must not overlap each other")
    ip, im = (1, 0) if z[0, 0, 0] > z[-1, 0, 0] else (0, 1)
    missing = default_fill(np.float64)
    for y in range(0, z.shape[1], block_rows):
        inputs = [np.asfortranarray(a[:, y:y + block_rows, :].T, dtype=np.float64) for a in arrays]
        scratch = [np.empty(inputs[0].shape[:2] + (len(levels),), dtype=np.float64, order='F')
                   for _ in range(3)]
        dinterp3dz_uvw(*inputs, levels, *scratch, missing, ip, im)
        for dest, values in zip(out, scratch):
            # Cast before matching fill values, as wrf.interplevel does.
            target = dest[:, y:y + block_rows, :]
            target[:] = values.T
            target[np.isclose(target, missing)] = np.nan
    return tuple(out)
