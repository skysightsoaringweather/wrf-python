# CAPE optimization regression fixture

`cape_reference.npz` contains six US model columns from the completed WRF frame
valid 2026-11-08 18 UTC (initialization 12 UTC). They span column-maximum CAPE
from zero to 2414 J/kg. Inputs are pressure in hPa, temperature in K, mixing
ratio in kg/kg, mass height and terrain in m, and surface pressure in hPa.
Production float32 inputs were promoted to float64 before calculation.

The `cape3d`/`cape2d` fields and their boolean masks were captured from the
existing forecast image's NCAR wrf-python 1.4.2, commit
`124a8336529af6397fe150e14bd436d923122cdd`, before installing the optimized fork.
The parent fork's original 3D Fortran was separately compiled and checked for
exact parity with this installed implementation during optimization review.

This is a regression reference for existing behavior, not independent validation
of the meteorological definition. `test/test_cape.py` uses it to check parcel
values/masks, floating-point types, vertical orientation, array layout, multiple
times, scalar columns, threading and the unchanged 2D wrapper path.
