"""CPU regression tests for the integer RDME propensity representation.

Run with: python3 -m unittest discover -s testing/host -v
Only a C++11 compiler is required; no CUDA toolkit, GPU, or LM install.
"""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class IntMpdTableTests(unittest.TestCase):
    def compile_and_run(self, source):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cpp = directory / "test.cpp"
            binary = directory / "test"
            cpp.write_text(source)
            subprocess.run(
                [os.environ.get("CXX", "g++"), "-std=c++11", "-O2", "-Wall",
                 "-Wextra", "-Werror", "-I", str(ROOT / "src"),
                 str(cpp), "-o", str(binary)], check=True,
            )
            subprocess.run([str(binary)], check=True)

    def test_sparse_lookup_matches_dense_reference(self):
        header = ROOT / "src/rdme/SecondOrderPropensityTable.h"
        self.assertTrue(header.is_file(), "The compact propensity table is not implemented")
        self.compile_and_run((ROOT / "testing/host/second_order_propensity_table.cpp").read_text())

    def test_dirty_reaction_model_is_uploaded_once(self):
        # Compile the real beginning of runTimestep. The CUDA upload itself is
        # replaced because this regression is about whether each pending model
        # is transferred once, and whether a failed upload remains pending.
        source = (ROOT / "src/rdme/IntMpdRdmeSolver.cu").read_text()
        body = source.split("void IntMpdRdmeSolver::runTimestep(", 1)[1]
        body = body.split("{", 1)[1].split("// Calculate some properties", 1)[0]
        self.compile_and_run(r'''
#include <cassert>
#include <stdexcept>
#define PROF_BEGIN(x) do {} while (0)
struct Solver {
    bool reactionModelModified = false;
    bool fail = false;
    unsigned uploads = 0;
    void copyModelsToDevice() {
        if (fail) throw std::runtime_error("upload failed");
        ++uploads;
    }
    void runTimestep() {''' + body + r'''}
};
int main() {
    Solver solver;
    solver.runTimestep();
    assert(solver.uploads == 0);
    solver.reactionModelModified = true;
    solver.runTimestep();
    solver.runTimestep();
    assert(solver.uploads == 1);
    solver.reactionModelModified = true;
    solver.fail = true;
    try { solver.runTimestep(); assert(false); }
    catch (const std::runtime_error&) {}
    assert(solver.reactionModelModified);
    solver.fail = false;
    solver.runTimestep();
    solver.runTimestep();
    assert(solver.uploads == 2);
}
''')

    def test_solver_compute_uses_the_compact_table(self):
        # Compile the actual host computation from the CUDA translation unit.
        # Small fixture types replace protobuf/CUDA-dependent base classes.
        source = (ROOT / "src/rdme/IntMpdRdmeSolver.cu").read_text()
        begin = source.index("void IntMpdRdmeSolver::computePropensities()")
        end = source.index("void IntMpdRdmeSolver::copyModelsToDevice()", begin)
        compute = source[begin:end].replace("IntMpdRdmeSolver::", "Solver::")
        fixture = (ROOT / "testing/host/int_mpd_propensities.cpp").read_text()
        self.compile_and_run(fixture.replace("// ACTUAL_SOLVER_COMPUTE", compute))

    def test_solver_cleanup_uses_owning_device_and_handles_partial_setup(self):
        source = (ROOT / "src/rdme/IntMpdRdmeSolver.cu").read_text()
        begin = source.index("IntMpdRdmeSolver::~IntMpdRdmeSolver()")
        end = source.index("void IntMpdRdmeSolver::allocateLattice(", begin)
        destructor = source[begin:end].replace("IntMpdRdmeSolver::~IntMpdRdmeSolver", "Solver::~Solver")
        destructor = destructor.replace("IntMpdRdmeSolver::", "Solver::")
        if "void Solver::freeDiffusionBuffers()" not in destructor:
            destructor += "void Solver::freeDiffusionBuffers() {}\nvoid Solver::freeReactionBuffers() {}\n"
        rebuild = source.split("void IntMpdRdmeSolver::buildDiffusionModel(", 1)[1]
        rebuild = rebuild.split("{", 1)[1].split("RDMESolver::buildDiffusionModel", 1)[0]
        fixture = (ROOT / "testing/host/int_mpd_cleanup.cpp").read_text()
        fixture = fixture.replace("// ACTUAL_DIFFUSION_REBUILD_PREFIX", rebuild)
        self.compile_and_run(fixture.replace("// ACTUAL_SOLVER_DESTRUCTOR", destructor))

    def test_overflow_allocation_holds_count_and_all_particle_pairs(self):
        for name in ("word_diffusion_1d_dev.cu", "byte_diffusion_1d_dev.cu",
                     "byte_diffusion_1d_mgpu_dev.cu"):
            source = (ROOT / "src/rdme/dev" / name).read_text()
            definition = next(line for line in source.splitlines()
                              if line.startswith("#define MPD_OVERFLOW_LIST_SIZE"))
            with self.subTest(source=name):
                self.compile_and_run("#include <cstdint>\n#include <cassert>\n"
                                     "#define TUNE_MPD_MAX_PARTICLE_OVERFLOWS 2048\n" + definition + r'''
int main() {
    uint32_t fullOverflowList[1 + 2 * TUNE_MPD_MAX_PARTICLE_OVERFLOWS];
    assert(MPD_OVERFLOW_LIST_SIZE == sizeof(fullOverflowList));
}
''')

    def test_word_reaction_evaluation_preserves_order_without_heap_allocation(self):
        source = (ROOT / "src/rdme/dev/word_reaction_dev.cu").read_text()
        marker = "\t#ifdef MPD_GLOBAL_S_MATRIX\n\tinline __device__ void evaluateReaction"
        evaluate = source[source.index(marker):]
        fixture = (ROOT / "testing/host/word_reaction_evaluation.cpp").read_text()
        for global_stoichiometry in (True, False):
            with self.subTest(global_stoichiometry=global_stoichiometry):
                code = fixture.replace("// ACTUAL_WORD_REACTION_EVALUATION", evaluate)
                if global_stoichiometry:
                    code = "#define MPD_GLOBAL_S_MATRIX\n" + code
                self.compile_and_run(code)


if __name__ == "__main__":
    unittest.main()
