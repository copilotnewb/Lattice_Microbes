#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include "rdme/SecondOrderPropensityTable.h"

using lm::rdme::SecondOrderPropensityTable;
using lm::rdme::SecondOrderTableView;

#define MPD_GLOBAL_S_MATRIX
#define MPD_GLOBAL_R_MATRIX
#define MPD_GLOBAL_T_MATRIX
#define CUDA_EXCEPTION_CHECK_NOTHROW(call) do { assert((call) == 0); } while (0)

static int currentDevice = 0;
static std::set<void*> allocations;
static unsigned int destroyedLattices = 0;
static void* allocate() {
    void* ptr = std::malloc(4);
    assert(ptr);
    allocations.insert(ptr);
    return ptr;
}
int cudaGetDevice(int* device) { *device = currentDevice; return 0; }
int cudaSetDevice(int device) { currentDevice = device; return 0; }
static int cudaFree(void* ptr) {
    assert(currentDevice == 0);
    assert(allocations.erase(ptr) == 1);
    std::free(ptr);
    return 0;
}
static int cudaStreamDestroy(void* stream) { return cudaFree(stream); }

struct Solver {
    int cudaDevice = -1;
    void *cudaOverflowList = NULL, *cudaStream = NULL;
    float *model_reactionRates = NULL, *zeroOrder = NULL, *firstOrder = NULL;
    void *propZeroOrder = NULL, *propFirstOrder = NULL, *propSecondOrder = NULL;
    void *secondOrderRowsG = NULL, *secondOrderColumnsG = NULL;
    void *RLG = NULL, *SG = NULL, *TG = NULL;
    void *reactionRatesG = NULL, *reactionOrdersG = NULL, *reactionSitesG = NULL;
    void *D1G = NULL, *D2G = NULL;
    void *lattice = NULL;
    size_t zeroOrderSize = 0, firstOrderSize = 0, secondOrderSize = 0;
    SecondOrderPropensityTable secondOrderTable;
    SecondOrderTableView secondOrderDeviceTable = {};
    void freeDiffusionBuffers();
    void freeReactionBuffers();
    void rebuildDiffusionState() {
        // ACTUAL_DIFFUSION_REBUILD_PREFIX
    }
    void destroyDiffusionModel() {
        if (lattice) { cudaFree(lattice); lattice = NULL; ++destroyedLattices; }
    }
    ~Solver();
};

// ACTUAL_SOLVER_DESTRUCTOR

int main() {
    // Destruction before initialize() must not initialize CUDA or free garbage.
    { Solver solver; }
    assert(allocations.empty());
    {
        Solver solver;
        solver.cudaDevice = 0;
        solver.cudaOverflowList = allocate();
        solver.cudaStream = allocate();
        solver.model_reactionRates = new float[2];
        solver.zeroOrder = new float[2];
        solver.firstOrder = new float[2];
        solver.propZeroOrder = allocate();
        solver.propFirstOrder = allocate();
        solver.propSecondOrder = allocate();
        solver.secondOrderRowsG = allocate();
        solver.secondOrderColumnsG = allocate();
        solver.RLG = allocate();
        solver.SG = allocate();
        solver.TG = allocate();
        solver.reactionRatesG = allocate();
        solver.reactionOrdersG = allocate();
        solver.reactionSitesG = allocate();
        solver.D1G = allocate();
        solver.D2G = allocate();
        solver.lattice = allocate();
        currentDevice = 1;
    }
    assert(allocations.empty());
    assert(destroyedLattices == 1);
    assert(currentDevice == 1);
    {
        // An allocation error partway through model setup leaves null buffers.
        Solver solver;
        solver.cudaDevice = 0;
        solver.secondOrderRowsG = allocate();
    }
    assert(allocations.empty());
    assert(currentDevice == 1);
    currentDevice = 0;
    {
        Solver solver;
        solver.cudaDevice = 0;
        solver.SG = allocate();
        void* reactionTopology = solver.SG;
        for (int sparse : {1, 0, 1}) {
            solver.rebuildDiffusionState();
            assert(solver.secondOrderRowsG == NULL);
            assert(solver.secondOrderColumnsG == NULL);
            assert(solver.propSecondOrder == NULL);
            assert(solver.SG == reactionTopology);
            assert(allocations.count(reactionTopology) == 1);
            solver.secondOrderTable.initialize(6, 3, {{0, 5}, {2, 2}}, sparse);
            solver.propSecondOrder = allocate();
            if (sparse) {
                solver.secondOrderRowsG = allocate();
                solver.secondOrderColumnsG = allocate();
            }
            solver.secondOrderDeviceTable = solver.secondOrderTable.view();
            solver.secondOrderDeviceTable.values = static_cast<float*>(solver.propSecondOrder);
            solver.secondOrderDeviceTable.rowOffsets = static_cast<unsigned int*>(solver.secondOrderRowsG);
            solver.secondOrderDeviceTable.columns = static_cast<unsigned int*>(solver.secondOrderColumnsG);
            assert((solver.secondOrderDeviceTable.rowOffsets != NULL) == (sparse != 0));
        }
    }
    assert(allocations.empty());
    std::cout << "PASS: owned device cleanup, current-device restoration, partial setup, and sparse/dense/sparse rebuild\n";
}
