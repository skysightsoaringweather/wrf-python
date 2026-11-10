"""Import must preserve explicitly configured OpenMP parallelism."""
import json
import os
import subprocess
import sys
import unittest


class OmpConfigTest(unittest.TestCase):
    def check_threads(self, setting, expected):
        env = os.environ.copy()
        # pandas may import numexpr, which cannot parse OpenMP thread lists.
        env["NUMEXPR_NUM_THREADS"] = "1"
        env.pop("OMP_NUM_THREADS", None)
        if setting is not None:
            env["OMP_NUM_THREADS"] = setting
        code = ("import json, wrf; print(json.dumps([wrf.omp_enabled(), "
                "wrf.omp_get_max_threads()]))")
        enabled, count = json.loads(subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True).strip())
        if not enabled:
            self.skipTest("Extension built without OpenMP")
        self.assertEqual(count, expected)

    def test_default_is_one(self):
        self.check_threads(None, 1)

    def test_explicit_threads(self):
        self.check_threads("4", 4)

    def test_nested_thread_list(self):
        self.check_threads("4,2", 4)


if __name__ == "__main__":
    unittest.main()
