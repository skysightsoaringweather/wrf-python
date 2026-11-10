"""CAPE regression cases captured before the column-workspace optimization.

The small fixture samples real US columns spanning inactive and buoyant parcels.
Double-precision tolerances allow compiler/libm rounding differences; missing
values and parcel ordering must match exactly. No external WRF file is needed.
"""
from pathlib import Path
import unittest

import numpy as np

from wrf import cape_2d, cape_3d, omp_get_max_threads, omp_set_num_threads


class CapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent / "ci_tests/cape_reference.npz") as f:
            cls.fixture = {name: f[name] for name in f.files}

    def setUp(self):
        self.threads = omp_get_max_threads()

    def tearDown(self):
        omp_set_num_threads(self.threads)

    def inputs(self, dtype=np.float64):
        return [self.fixture[n].astype(dtype) for n in
                ("p", "t", "q", "z", "terrain", "surface")]

    def check_result(self, actual, name, index=None):
        expected = self.fixture[name]
        mask = self.fixture[name + "_mask"]
        if index is not None:
            expected, mask = expected[index], mask[index]
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), mask)
        tolerance = 2e-4 if actual.dtype == np.float32 else 1e-9
        np.testing.assert_allclose(np.ma.getdata(actual)[~mask],
                                   expected[~mask], rtol=1e-12, atol=tolerance)

    def test_full_parcel_values_and_threads(self):
        for threads in (1, 4):
            omp_set_num_threads(threads)
            for dtype in (np.float64, np.float32):
                with self.subTest(threads=threads, dtype=dtype):
                    self.check_result(cape_3d(*self.inputs(dtype), True,
                                             meta=False), "cape3d")

    def test_reversed_vertical_order(self):
        inputs = self.inputs()
        inputs[:4] = [a[::-1] for a in inputs[:4]]
        self.check_result(cape_3d(*inputs, True, meta=False)[:, ::-1], "cape3d")

    def test_noncontiguous_horizontal_views(self):
        inputs = [np.repeat(a, 2, axis=-1)[..., ::2] for a in self.inputs()]
        self.check_result(cape_3d(*inputs, True, meta=False), "cape3d")

    def test_scalar_column(self):
        inputs = [a[:, -1, -1] for a in self.inputs()[:4]]
        inputs += [float(a[-1, -1]) for a in self.inputs()[4:]]
        result = cape_3d(*inputs, True, meta=False)[:, :, 0, 0]
        self.check_result(result, "cape3d", (slice(None), slice(None), -1, -1))

    def test_multiple_times(self):
        inputs = [np.stack([a, a]) for a in self.inputs()]
        result = cape_3d(*inputs, True, meta=False)
        for time in (0, 1):
            self.check_result(result[:, time], "cape3d")

    def test_2d_workspace_path_is_unchanged(self):
        self.check_result(cape_2d(*self.inputs(), True, meta=False), "cape2d")


if __name__ == "__main__":
    unittest.main()
