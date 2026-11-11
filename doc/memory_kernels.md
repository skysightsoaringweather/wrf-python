# Bounded-memory CAPE and wind profiles

`wrf.cape_3d_max(pres_hpa, tkel, qv, height, block_rows=16)` returns the vertical maximum of the existing all-parcel 3D CAPE calculation. It does not change the parcel choice to surface, mixed-layer, or most-unstable 2D CAPE. The shared Fortran core retains the existing arithmetic, skips missing parcel results, and retains the valid zero for the top parcel. The result has the pressure dtype and default CAPE masking convention. The original `cape_3d` and `cape_2d` APIs remain available and unchanged.

Inputs must be nonempty float32/float64 `(level, y, x)` arrays for one grid with no missing/nonfinite values. Either vertical order and strided arrays are supported. Terrain and surface pressure are not dependencies of the all-parcel algorithm. The wrapper converts only one row block to double precision at a time; Fortran returns one CAPE value per column, without full CAPE or CIN volumes. New API calls serialize access to the existing fixed-unit pseudoadiabat file reader while retaining native OpenMP parallelism.

`wrf.interplevel_uvw(u, v, w, height, levels, out=None, block_rows=16)` shares one bracket search and double-precision interpolation weights across three collocated components. Output is a tuple of `(len(levels), y, x)` arrays, with NaNs for missing values. Supplying three writable output buffers fills them directly, including strided destinations; buffers must not overlap inputs or each other. Each component retains its dtype.

Both kernels preserve their branch's existing scientific behavior. In particular, the `optim` fork includes the upper interpolation endpoint, whereas the NCAR-compatible image branch requires strict inequalities. The combined wind kernel follows the existing `interplevel` in each branch, including reversed grids, top-down searches through inversions, and repeated heights. The separate 2D CAPE parcel search also remains branch-specific.

Regression commands:

```sh
python -m unittest discover -s test -p test_cape.py -v
python -m unittest discover -s test -p test_memory_kernels.py -v
```

The memory suite compares against the established public APIs for float32/64, one/four OpenMP threads, several row-block sizes, reversed/strided arrays, and edge cases. The existing CAPE fixtures additionally guard against regressions shared by the full and maximum-only paths. Tested with Python 3.12 / NumPy 1.26 and the forecast image's Python 3.14 / NumPy 2.3 runtimes.
