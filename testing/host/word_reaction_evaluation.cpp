#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define __device__
#define __CUDA_ARCH__ 750
#define MPD_PARTICLES_PER_SITE 16
#define TUNE_MPD_MAX_PARTICLE_OVERFLOWS 4
#define read_element(array, index) ((array)[index])
typedef unsigned int uint;
static unsigned int numberSpeciesC = 0;
#ifndef MPD_GLOBAL_S_MATRIX
static int8_t* SC = NULL;
#endif
static unsigned int heapAllocations = 0;
void* trackedMalloc(size_t bytes) { ++heapAllocations; return std::malloc(bytes); }
void trackedFree(void* ptr) { std::free(ptr); }
unsigned int atomicAdd(unsigned int* value, unsigned int amount) {
    const unsigned int old = *value;
    *value += amount;
    return old;
}

#define malloc trackedMalloc
#define free trackedFree
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
// ACTUAL_WORD_REACTION_EVALUATION
#pragma GCC diagnostic pop
#undef malloc
#undef free

// Independent reference: the original code copied each reaction's full S
// column, consumed negative entries one particle at a time, then appended
// positive species in ascending order. Valid reactions have enough reactants.
static void reference(unsigned int* particles, const std::vector<int8_t>& changes,
                      unsigned int* overflow, unsigned int latticeIndex) {
    std::vector<int8_t> remaining(changes);
    unsigned int next = 0;
    for (unsigned int i = 0; i < 16; ++i) {
        const unsigned int particle = particles[i];
        if (particle == 0) continue;
        if (remaining[particle - 1] >= 0) particles[next++] = particle;
        else ++remaining[particle - 1];
    }
    for (unsigned int species = 0; species < remaining.size(); ++species) {
        assert(remaining[species] >= 0);
        for (int j = 0; j < remaining[species]; ++j) {
            if (next < 16) particles[next++] = species + 1;
            else {
                const unsigned int index = overflow[0]++;
                if (index < TUNE_MPD_MAX_PARTICLE_OVERFLOWS) {
                    overflow[2 * index + 1] = latticeIndex;
                    overflow[2 * index + 2] = species + 1;
                }
            }
        }
    }
    while (next < 16) particles[next++] = 0;
}

static void check(const std::vector<unsigned int>& input,
                  const std::vector<int8_t>& changes, unsigned int existingOverflows) {
    unsigned int actual[16] = {}, expected[16] = {};
    std::copy(input.begin(), input.end(), actual);
    std::copy(input.begin(), input.end(), expected);
    unsigned int actualOverflow[1 + 2 * TUNE_MPD_MAX_PARTICLE_OVERFLOWS];
    unsigned int expectedOverflow[1 + 2 * TUNE_MPD_MAX_PARTICLE_OVERFLOWS];
    std::fill(actualOverflow, actualOverflow + 9, 12345);
    std::fill(expectedOverflow, expectedOverflow + 9, 12345);
    actualOverflow[0] = expectedOverflow[0] = existingOverflows;
    numberSpeciesC = changes.size();
    // Use reaction index 1 to exercise the full-species stride, not just row 0.
    std::vector<int8_t> stoichiometry(numberSpeciesC * 2, 0);
    std::copy(changes.begin(), changes.end(), stoichiometry.begin() + numberSpeciesC);
#ifdef MPD_GLOBAL_S_MATRIX
    evaluateReaction(982, 2, actual, 1, actualOverflow, stoichiometry.data());
#else
    SC = stoichiometry.data();
    evaluateReaction(982, 2, actual, 1, actualOverflow);
#endif
    reference(expected, changes, expectedOverflow, 982);
    assert(std::equal(actual, actual + 16, expected));
    assert(std::equal(actualOverflow, actualOverflow + 9, expectedOverflow));
    assert(heapAllocations == 0);
}

int main() {
    std::vector<int8_t> changes(1024, 0);
    changes[299] = -1; changes[1] = -1; changes[8] = 2;
    check({300, 2, 300, 7, 0, 300}, changes, 0);
    std::fill(changes.begin(), changes.end(), 0);
    changes[0] = -3; changes[999] = 2;
    check({1, 1, 8, 1, 9, 1}, changes, 1);
    std::fill(changes.begin(), changes.end(), 0);
    changes[0] = 3; changes[499] = 4;
    check(std::vector<unsigned int>(16, 1024), changes, 0);
    check(std::vector<unsigned int>(16, 1024), changes, 3);
    std::fill(changes.begin(), changes.end(), 0);
    changes[1023] = -16; changes[299] = 17;
    check(std::vector<unsigned int>(16, 1024), changes, 0);
    std::fill(changes.begin(), changes.end(), 0);
    check({0, 1024, 0, 1, 0, 7}, changes, 0);
    check({}, changes, 0);

    std::mt19937 rng(19837);
    for (unsigned int trial = 0; trial < 5000; ++trial) {
        std::vector<unsigned int> input(16);
        std::vector<unsigned int> available(1024, 0);
        std::fill(changes.begin(), changes.end(), 0);
        for (unsigned int& particle : input) {
            particle = rng() % 6 == 0 ? 0 : (rng() % 8 + 1) * 100;
            if (particle) ++available[particle - 1];
        }
        for (unsigned int i = 0; i < available.size(); ++i)
            if (available[i]) changes[i] = -static_cast<int>(rng() % (available[i] + 1));
        for (unsigned int i = 0; i < 4; ++i) {
            unsigned int species = rng() % 1024;
            if (changes[species] == 0) changes[species] = rng() % 4;
        }
        check(input, changes, rng() % 5);
    }
    std::cout << "PASS: actual word reaction evaluation, 5000 randomized stoichiometries, particle/overflow order, and no heap allocation\n";
}
