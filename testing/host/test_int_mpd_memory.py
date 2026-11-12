"""Exercise the read-only audit on the actual LM HDF5 schema."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = None


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "utils/int_mpd_memory.py"


@unittest.skipIf(h5py is None, "The HDF5 audit tests require numpy and h5py")
class IntMpdMemoryTests(unittest.TestCase):
    def write_model(self, filename):
        with h5py.File(filename, "w") as file:
            diffusion = file.create_group("Model/Diffusion")
            diffusion.attrs.update(numberSpecies=6, numberReactions=5, numberSiteTypes=3,
                                   latticeXSize=64, latticeYSize=64, latticeZSize=128,
                                   particlesPerSite=16, bytes_per_particle=4)
            reactions = file.create_group("Model/Reaction")
            reactions.attrs.update(numberSpecies=6, numberReactions=5)
            reactions.create_dataset("ReactionTypes", data=[0, 1, 2, 2, 3], dtype="u4")
            dependencies = np.zeros((6, 5), dtype="u4")
            dependencies[0, 1] = 1
            dependencies[[0, 5], 2] = 1
            dependencies[[0, 5], 3] = 1
            dependencies[2, 4] = 1
            reactions.create_dataset("DependencyMatrix", data=dependencies)

    def test_hdf5_audit_deduplicates_pairs_and_includes_self_reactions(self):
        self.assertTrue(SCRIPT.is_file(), "The finalized-model memory audit is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "fixture.lm"
            self.write_model(filename)
            before = filename.read_bytes()
            result = subprocess.run([sys.executable, str(SCRIPT), str(filename), "--json"],
                                    text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["species"], 6)
            self.assertEqual(report["reactions"], 5)
            self.assertEqual(report["site_types"], 3)
            self.assertEqual(report["unique_second_order_pairs"], 2)
            self.assertEqual(report["active_second_order_species"], 3)
            self.assertEqual(report["second_order_dense_bytes"], 432)
            self.assertEqual(report["second_order_sparse_bytes"], 60)
            self.assertEqual(report["device_arrays_bytes"]["particle_lattices"], 67108864)
            self.assertEqual(report["device_arrays_bytes"]["site_lattice"], 524288)
            self.assertEqual(report["second_order_saved_bytes"], 372)
            self.assertEqual(before, filename.read_bytes())

    def test_bad_second_order_dependency_count_is_rejected(self):
        self.assertTrue(SCRIPT.is_file(), "The finalized-model memory audit is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "fixture.lm"
            self.write_model(filename)
            with h5py.File(filename, "r+") as file:
                file["Model/Reaction/DependencyMatrix"][5, 2] = 0
            result = subprocess.run([sys.executable, str(SCRIPT), str(filename)],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reaction 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
