#include "rdme/SecondOrderPropensityTable.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using lm::rdme::SecondOrderPropensityTable;
using lm::rdme::SecondOrderTableView;

static bool sameBits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

struct Reaction {
    unsigned int a, b;
    double rate;
    bool self;
    unsigned int siteMask;
};

static void compareWithOriginalDense(const unsigned int species,
                                    const unsigned int sites,
                                    const std::vector<Reaction>& reactions) {
    std::vector<SecondOrderPropensityTable::SpeciesPair> pairs;
    for (const Reaction& reaction : reactions)
        pairs.push_back(std::make_pair(reaction.a, reaction.b));

    SecondOrderPropensityTable sparse, dense;
    sparse.initialize(species, sites, pairs, true);
    dense.initialize(species, sites, pairs, false);
    std::vector<float> reference(size_t(species) * species * sites, 0.0f);
    const double tau = 50e-6;
    const float scale = 64 * 64 * 128;

    // Keep the original dense assignments, expression precision, site order,
    // reaction order and self-reaction factor. Duplicate unlike-species
    // reactions intentionally overwrite here, matching the existing solver.
    for (unsigned int site = 0; site < sites; ++site) {
        for (const Reaction& reaction : reactions) {
            if (!(reaction.siteMask & (1U << site))) continue;
            const size_t ab = size_t(site) * species * species + reaction.a * species + reaction.b;
            const size_t ba = size_t(site) * species * species + reaction.b * species + reaction.a;
            if (reaction.self) {
                reference[ab] += reaction.rate * tau * scale * 2;
                sparse.add(site, reaction.a, reaction.b, reaction.rate * tau * scale * 2);
                dense.add(site, reaction.a, reaction.b, reaction.rate * tau * scale * 2);
            } else {
                reference[ab] = reaction.rate * tau * scale;
                reference[ba] = reaction.rate * tau * scale;
                sparse.set(site, reaction.a, reaction.b, reaction.rate * tau * scale);
                dense.set(site, reaction.a, reaction.b, reaction.rate * tau * scale);
            }
        }
    }

    const SecondOrderTableView sparseView = sparse.view();
    const SecondOrderTableView denseView = dense.view();
    for (unsigned int site = 0; site < sites; ++site) {
        assert(sparseView.lookup(site, 0, 1) == 0.0f);
        assert(sparseView.lookup(site, 1, 0) == 0.0f);
        for (unsigned int a = 0; a < species; ++a)
            for (unsigned int b = 0; b < species; ++b) {
                const float want = reference[size_t(site) * species * species + a * species + b];
                assert(sameBits(want, sparseView.lookup(site, a + 1, b + 1)));
                assert(sameBits(want, denseView.lookup(site, a + 1, b + 1)));
            }
    }

    // Exercise the production lookup using the kernel's actual particle loop,
    // including empty slots, repeated particles, and the largest species ID.
    std::mt19937 rng(209812);
    for (unsigned int trial = 0; trial < 1000; ++trial) {
        unsigned int particles[16];
        for (unsigned int& particle : particles) particle = rng() % (species + 1);
        const unsigned int site = rng() % sites;
        float actual = 0.25f, expected = 0.25f;
        for (unsigned int i = 0; i < 16; ++i) {
            const unsigned int p1 = particles[i];
            if (p1 > 0) {
                for (unsigned int j = i + 1; j < 16; ++j) {
                    const unsigned int p2 = particles[j];
                    if (p2 > 0) {
                        expected += reference[size_t(site) * species * species + (p1 - 1) * species + (p2 - 1)];
                        actual += sparseView.lookup(site, p1, p2);
                    }
                }
            }
        }
        assert(sameBits(actual, expected));
    }

    // A rate update must overwrite stale nonzero data, including a transition
    // to zero, without rebuilding or losing any potential reactive pair.
    sparse.clear();
    dense.clear();
    for (unsigned int site = 0; site < sites; ++site)
        for (const Reaction& reaction : reactions) {
            assert(sparse.view().lookup(site, reaction.a + 1, reaction.b + 1) == 0.0f);
            assert(dense.view().lookup(site, reaction.a + 1, reaction.b + 1) == 0.0f);
        }
}

int main() {
    compareWithOriginalDense(6, 3, {
        {0, 5, 0.1, false, 1}, {5, 0, 0.7, false, 1},
        {2, 2, 0.2, true, 3}, {2, 2, 0.000000001, true, 3},
        {2, 2, 0.5, false, 1}, {2, 2, 0.3, true, 1},
        {1, 3, 0.0, false, 7}, {1, 3, 0.2, false, 2},
        {0, 0, 0.123456789123, true, 4},
    });
    compareWithOriginalDense(1, 1, {});

    std::mt19937 rng(12003);
    std::vector<Reaction> reactions;
    for (unsigned int i = 0; i < 500; ++i) {
        unsigned int a = rng() % 31, b = rng() % 31;
        const bool self = rng() % 3 == 0;
        if (self) b = a;
        reactions.push_back({a, b, (rng() % 100000) * 1e-8, self, static_cast<unsigned int>(rng() % 16)});
    }
    compareWithOriginalDense(31, 4, reactions);

    // This shape previously needed 280 GB of values. Verify sparse storage
    // scales with the pair topology, and size arithmetic exceeds 32 bits.
    SecondOrderPropensityTable large;
    large.initialize(100000, 7, {{0, 99999}, {99999, 0}, {4, 4}}, true);
    assert(large.valueCount() == 14);
    assert(large.storageBytes() < 500000);
    assert(large.denseBytes() == size_t(280000000000ULL));
    large.set(6, 0, 99999, 1.0);
    assert(large.view().lookup(6, 100000, 1) == 1.0f);

    bool invalid = false;
    try { large.initialize(2, 1, {{0, 2}}, true); }
    catch (const std::out_of_range&) { invalid = true; }
    assert(invalid);
    bool overflow = false;
    try { SecondOrderPropensityTable::checkedProduct(std::numeric_limits<size_t>::max(), 2); }
    catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);

    std::cout << "PASS: dense/sparse lookup, duplicate and self reactions, particle order, updates, and allocation arithmetic\n";
}
