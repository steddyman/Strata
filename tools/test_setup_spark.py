"""Tests for setup.py on an NVIDIA DGX Spark (GB10, arm64 Linux, unified memory) with mocked nvidia-smi, lscpu and
/proc/meminfo answers: the GPU is found although nvidia-smi gives its memory as [N/A], its memory is counted as Strix
Halo's is (all of the RAM but the OS's share, "uma"), the arm64 CPU is named and needs no x86 ISA floor, and a discrete
card on an x86 PC is unchanged.  No GPU, no downloads.

    python -m unittest tools.test_setup_spark
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

SPARK_SMI = "0, NVIDIA GB10, [N/A], 12.1, 580.95.05\n"
LSCPU = ("Architecture:             aarch64\n"
         "Vendor ID:                ARM\n"
         "  Model name:             Cortex-X925\n"
         "  Model name:             Cortex-A725\n")


class Spark(unittest.TestCase):
    def setUp(self):
        self.arm = setup.ARM64_LINUX
        setup.ARM64_LINUX = True

    def tearDown(self):
        setup.ARM64_LINUX = self.arm

    def test_gb10_found_with_unified_memory(self):
        with mock.patch.object(setup, "out", return_value=SPARK_SMI), mock.patch.object(setup, "ram_gb", return_value=119.7):
            found = setup.gpus()
        self.assertEqual(len(found), 1)
        g = found[0]
        self.assertEqual((g["index"], g["name"], g["arch"]), (0, "NVIDIA GB10", "121"))
        self.assertTrue(g["uma"])
        self.assertEqual(g["dedicated_gb"], 0.0)
        self.assertAlmostEqual(g["vram_gb"], 119.7 - setup.UMA_OS_LEFT_GB)
        self.assertEqual(setup.low_ram_vram(g), 0.0)        # its memory is the RAM: nothing beside it

    def test_cpu_named_and_no_floor(self):
        with mock.patch.object(setup, "out", return_value=LSCPU):
            name, avx2, avx512 = setup.cpu_info()
        self.assertEqual(name, "Cortex-X925 + Cortex-A725")
        self.assertFalse(avx2 or avx512)
        self.assertEqual(setup.cpu_floor(False), "")

    def test_x86_card_unchanged(self):
        setup.ARM64_LINUX = False
        with mock.patch.object(setup, "out", return_value="0, NVIDIA GeForce RTX 5090, 32607, 12.0, 610.57\n"):
            g = setup.gpus()[0]
        self.assertNotIn("uma", g)
        self.assertAlmostEqual(g["vram_gb"], 32607 / 1024.0)

    def test_deepseek_plan_counts_unified_memory_once(self):
        # the GB10's experts are the RAM: no "fits" that counts the same memory twice; the rest is read from the SSD
        from strata_control import dsv41_fit
        g = {"index": 0, "name": "NVIDIA GB10", "vram_gb": 115.7, "arch": "121", "uma": True, "dedicated_gb": 0.0}
        p = dsv41_fit.plan([g], 121.7, 3300, 8192, model="IST-DAS-2.00bpw")
        self.assertEqual(p["verdict"], "tight")
        self.assertEqual(p["spare_ram_gb"], 0.0)
        self.assertLess(p["vram_experts_gb"], 115.7)                 # the engine leaves the OS its share (measured: 106.8 GB)
        self.assertAlmostEqual(p["vram_experts_gb"] + p["cpu_experts_gb"], 132.1, delta=0.2)
        self.assertTrue(any("unified memory" in line for line in p["lines"]))
        d = dsv41_fit.plan([{**g, "uma": False}], 121.7, 3300, 8192, model="IST-DAS-2.00bpw")
        self.assertEqual(d["verdict"], "fits")                        # a discrete card of that size: the old rule

    def test_na_memory_skipped_off_arm64(self):
        setup.ARM64_LINUX = False                           # an x86 PC keeps the old rule: a row it cannot read is skipped
        with mock.patch.object(setup, "out", return_value=SPARK_SMI):
            self.assertEqual(setup.gpus(), [])


if __name__ == "__main__":
    unittest.main()
