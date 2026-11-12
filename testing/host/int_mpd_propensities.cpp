#include "rdme/SecondOrderPropensityTable.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using lm::rdme::SecondOrderPropensityTable;
typedef unsigned int uint;

struct ZerothOrderPropensityArgs { enum { REACTION_TYPE = 0 }; double k; };
struct FirstOrderPropensityArgs { enum { REACTION_TYPE = 1 }; double k; uint si; };
struct SecondOrderPropensityArgs { enum { REACTION_TYPE = 2 }; double k; uint s1i, s2i; };
struct SecondOrderSelfPropensityArgs { enum { REACTION_TYPE = 3 }; double k; uint si; };
struct InvalidArgException : std::runtime_error {
    InvalidArgException(const char*, const char* message, uint) : std::runtime_error(message) {}
};
struct Lattice {
    uint getXSize() { return 64; }
    uint getYSize() { return 64; }
    uint getZSize() { return 128; }
};

struct Solver {
    Lattice modelLattice;
    Lattice* lattice = &modelLattice;
    uint numberSpecies = 6, numberSiteTypes = 3, numberReactions = 8;
    double tau = 50e-6;
    std::vector<uint> reactionTypes, RL;
    std::vector<void*> propensityFunctionArgs;
    std::vector<float> model_reactionRates, zeroOrder, firstOrder;
    // Retain legacy storage in the test fixture so this regression also runs
    // against the original function, which leaves the compact table empty.
    std::vector<float> secondOrder;
    SecondOrderPropensityTable secondOrderTable;
    void computePropensities();
};

// ACTUAL_SOLVER_COMPUTE

static bool sameBits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

int main() {
    for (int sparse = 0; sparse <= 1; ++sparse) {
        Solver solver;
        ZerothOrderPropensityArgs r0 = {0.123456789012};
        FirstOrderPropensityArgs r1 = {0.234567890123, 0};
        SecondOrderPropensityArgs r2 = {0.345678901234, 0, 5};
        SecondOrderPropensityArgs r3 = {0.456789012345, 5, 0};
        SecondOrderSelfPropensityArgs r4 = {0.567890123456, 2};
        SecondOrderSelfPropensityArgs r5 = {0.000000000001, 2};
        SecondOrderPropensityArgs r6 = {0.0, 1, 3};
        SecondOrderSelfPropensityArgs r7 = {0.00123456789, 5};
        solver.reactionTypes = {0, 1, 2, 2, 3, 3, 2, 3};
        solver.propensityFunctionArgs = {&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7};
        solver.RL = {1,1,0, 1,0,1, 1,1,0, 1,0,0, 0,1,0, 0,1,0, 1,1,1, 1,0,0};
        solver.model_reactionRates.resize(solver.numberReactions);
        solver.zeroOrder.resize(3);
        solver.firstOrder.resize(18);
        solver.secondOrder.resize(108);
        solver.secondOrderTable.initialize(6, 3, {{0,5}, {2,2}, {1,3}, {5,5}}, sparse);

        for (int update = 0; update < 2; ++update) {
            if (update) { r3.k = 0.0; r4.k = 0.0; r5.k = 0.0; r6.k = 0.123456789012; }
            solver.computePropensities();
            std::vector<float> expected(108, 0.0f);
            const float scale = 64 * 64 * 128;
            expected[0 * 36 + 0 * 6 + 5] = r3.k * solver.tau * scale;
            expected[0 * 36 + 5 * 6 + 0] = r3.k * solver.tau * scale;
            expected[1 * 36 + 0 * 6 + 5] = r2.k * solver.tau * scale;
            expected[1 * 36 + 5 * 6 + 0] = r2.k * solver.tau * scale;
            expected[1 * 36 + 2 * 6 + 2] += r4.k * solver.tau * scale * 2;
            expected[1 * 36 + 2 * 6 + 2] += r5.k * solver.tau * scale * 2;
            expected[0 * 36 + 5 * 6 + 5] += r7.k * solver.tau * scale * 2;
            for (uint site = 0; site < 3; ++site) {
                expected[site * 36 + 1 * 6 + 3] = r6.k * solver.tau * scale;
                expected[site * 36 + 3 * 6 + 1] = r6.k * solver.tau * scale;
                for (uint a = 0; a < 6; ++a)
                    for (uint b = 0; b < 6; ++b)
                        assert(sameBits(expected[site * 36 + a * 6 + b],
                                        solver.secondOrderTable.view().lookup(site, a + 1, b + 1)));
            }
            assert(sameBits(solver.zeroOrder[0], r0.k * solver.tau / scale));
            assert(sameBits(solver.zeroOrder[1], r0.k * solver.tau / scale));
            assert(solver.zeroOrder[2] == 0.0f);
            assert(sameBits(solver.firstOrder[0], r1.k * solver.tau));
            assert(sameBits(solver.firstOrder[12], r1.k * solver.tau));
            assert(solver.firstOrder[6] == 0.0f);
            assert(sameBits(solver.model_reactionRates[3], r3.k * solver.tau * 64 * 64 * 128));
            assert(sameBits(solver.model_reactionRates[4], r4.k * solver.tau * 64 * 64 * 128));
        }
    }
    std::cout << "PASS: actual solver host propensity calculation and subsequent rate update in dense and sparse modes\n";
}
