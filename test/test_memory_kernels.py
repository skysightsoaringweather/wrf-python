"""Compare bounded kernels against the established public algorithms."""
from pathlib import Path
import unittest

import numpy as np
from wrf import cape_3d, cape_3d_max, interplevel, interplevel_uvw, omp_get_max_threads, omp_set_num_threads


class MemoryKernelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent / 'ci_tests/cape_reference.npz') as f:
            cls.inputs = [f[n] for n in ('p', 't', 'q', 'z', 'terrain', 'surface')]

    def setUp(self):
        self.threads = omp_get_max_threads()

    def tearDown(self):
        omp_set_num_threads(self.threads)

    def check_cape(self, fields, rows):
        expected = np.max(cape_3d(*fields, True, meta=False)[0], axis=0)
        actual = cape_3d_max(*fields[:4], block_rows=rows)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_array_equal(actual.data, expected.data)

    def test_cape_real_columns_precision_order_strides_and_threads(self):
        for threads in (1, 4):
            omp_set_num_threads(threads)
            for dtype in (np.float32, np.float64):
                fields = [a.astype(dtype) for a in self.inputs]
                for rows in (1, 2, 16):
                    for reverse in (False, True):
                        with self.subTest(threads=threads, dtype=dtype, rows=rows, reverse=reverse):
                            view = [a[..., ::-1] for a in fields]
                            if reverse:
                                view[:4] = [a[::-1] for a in view[:4]]
                            self.check_cape(view, rows)

    def test_cape_short_dry_saturated_low_pressure_columns(self):
        for n in (1, 2, 3, 20):
            for moisture in (0., .002, .025):
                p = np.broadcast_to(np.linspace(1000, 40, n)[:, None, None], (n, 3, 4)).copy()
                z = np.broadcast_to(np.linspace(100, 18000, n)[:, None, None], p.shape).copy()
                t = np.clip(300 - z*.006, 195, 310)
                q = np.full_like(p, moisture)
                self.check_cape([p, t, q, z, z[0], p[0]], 2)

    def test_cape_rejects_unsupported_inputs(self):
        fields = [a.astype(np.float32) for a in self.inputs[:4]]
        for rows in (0, -1):
            with self.assertRaises(ValueError): cape_3d_max(*fields, block_rows=rows)
        invalid = fields[0].copy(); invalid.flat[0] = np.nan
        with self.assertRaises(ValueError): cape_3d_max(invalid, *fields[1:])
        masked = np.ma.array(fields[0], mask=False)
        masked.mask.flat[0] = True
        with self.assertRaises(ValueError): cape_3d_max(masked, *fields[1:])
        with self.assertRaises(ValueError): cape_3d_max(fields[0][:-1], *fields[1:])
        with self.assertRaises(ValueError): cape_3d_max(*(a[0] for a in fields))

    def test_wind_preserves_brackets_missing_values_and_precision(self):
        rng = np.random.default_rng(22)
        base_z = np.array([100, 610, 1220, 1800, 2440, 4000, 8000], dtype=np.float64)
        levels = np.array([-50, 100, 610, 611, 1220, 2000, 2440, 8000, 9000], dtype=np.float32)
        for threads in (1, 4):
            omp_set_num_threads(threads)
            for dtype in (np.float32, np.float64):
                z = np.broadcast_to(base_z[:, None, None], (7, 5, 6)).astype(dtype).copy()
                z[:, 1, 2] += 325
                z[2:5, 2, 2] = [2000, 1300, 2500]  # Inversion: preserve top-down search.
                z[3, 3, 3] = z[2, 3, 3]  # Repeated heights.
                z[3, 4, 4] = np.nan
                fields = [rng.uniform(-150, 150, z.shape).astype(dtype) for _ in range(3)]
                fields[1][3, 0, 1] = np.nan
                for reverse in (False, True):
                    for rows in (1, 2, 16):
                        with self.subTest(threads=threads, dtype=dtype, reverse=reverse, rows=rows):
                            values = [a[::-1, :, ::-1] if reverse else a for a in (*fields, z)]
                            expected = [np.asarray(np.ma.filled(interplevel(a, values[3], levels, meta=False), np.nan), dtype=dtype)
                                        for a in values[:3]]
                            storage = [np.empty((len(levels), 10, 6), dtype=dtype) for _ in range(3)]
                            outputs = tuple(a[:, ::2] for a in storage)
                            actual = interplevel_uvw(*values, levels, out=outputs, block_rows=rows)
                            for result, ref, dest in zip(actual, expected, outputs):
                                self.assertIs(result, dest)
                                np.testing.assert_array_equal(result, ref)

    def test_wind_single_level_and_rejects_overlapping_outputs(self):
        z = np.ones((1, 2, 3), dtype=np.float32)
        outputs = interplevel_uvw(z, z, z, z, [0, 1, 2])
        self.assertTrue(all(np.isnan(a).all() for a in outputs))
        with self.assertRaises(ValueError): interplevel_uvw(z, z, z, z, [1], out=(z, z, z))
        with self.assertRaises(ValueError): interplevel_uvw(z, z, z, z, [])


if __name__ == '__main__': unittest.main()
