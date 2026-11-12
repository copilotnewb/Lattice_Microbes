/*
 * Compact storage for the integer RDME solver's precomputed pair propensities.
 * Distributed under the University of Illinois Open Source License used by
 * Lattice Microbes. See docs/LICENSE.
 */

#ifndef LM_RDME_SECONDORDERPROPENSITYTABLE_H_
#define LM_RDME_SECONDORDERPROPENSITYTABLE_H_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#define LM_SECOND_ORDER_HD __host__ __device__
#else
#define LM_SECOND_ORDER_HD
#endif

namespace lm {
namespace rdme {

template <typename T>
LM_SECOND_ORDER_HD inline T readSecondOrderElement(const T* values, size_t index)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(values + index);
#else
    return values[index];
#endif
}

// POD passed by value to the kernel. A null rowOffsets selects the original
// dense layout; otherwise rows contain sorted, unique canonical pairs (a<=b).
// Particle identifiers are one-based, with zero denoting an unoccupied slot.
struct SecondOrderTableView
{
    const float* values;
    const unsigned int* rowOffsets;
    const unsigned int* columns;
    size_t valuesPerSite;
    unsigned int numberSpecies;
    unsigned int numberSiteTypes;

    LM_SECOND_ORDER_HD float lookup(unsigned int site, unsigned int particle1,
                                   unsigned int particle2) const
    {
        if (particle1 == 0 || particle2 == 0 || particle1 > numberSpecies ||
            particle2 > numberSpecies || site >= numberSiteTypes)
            return 0.0f;

        unsigned int a = particle1 - 1, b = particle2 - 1;
        if (rowOffsets == NULL)
            return readSecondOrderElement(values, size_t(site) * valuesPerSite +
                                          size_t(a) * numberSpecies + b);

        if (a > b)
        {
            const unsigned int swap = a;
            a = b;
            b = swap;
        }
        unsigned int first = readSecondOrderElement(rowOffsets, a);
        const unsigned int end = readSecondOrderElement(rowOffsets, size_t(a) + 1);
        unsigned int last = end;
        while (first < last)
        {
            const unsigned int middle = first + (last - first) / 2;
            if (readSecondOrderElement(columns, middle) < b)
                first = middle + 1;
            else
                last = middle;
        }
        if (first == end || readSecondOrderElement(columns, first) != b)
            return 0.0f;
        return readSecondOrderElement(values, size_t(site) * valuesPerSite + first);
    }
};

// The topology is independent of rates, including rates that are initially
// zero. Recomputing rates therefore needs only clear/set/add and one value
// transfer; the CSR metadata is uploaded once during model construction.
class SecondOrderPropensityTable
{
public:
    typedef std::pair<unsigned int, unsigned int> SpeciesPair;

    SecondOrderPropensityTable()
        : species_(0), sites_(0), sparse_(true), valuesPerSite_(0), denseBytes_(0)
    {
    }

    static size_t checkedProduct(size_t a, size_t b)
    {
        if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
            throw std::overflow_error("Second-order propensity table size exceeds size_t");
        return a * b;
    }

    void initialize(unsigned int species, unsigned int sites,
                    const std::vector<SpeciesPair>& reactionPairs, bool sparse)
    {
        species_ = species;
        sites_ = sites;
        sparse_ = sparse;
        denseBytes_ = checkedProduct(checkedProduct(checkedProduct(species, species), sites), sizeof(float));
        rows_.clear();
        columns_.clear();
        values_.clear();

        std::vector<SpeciesPair> pairs(reactionPairs);
        for (SpeciesPair& pair : pairs)
        {
            if (pair.first >= species || pair.second >= species)
                throw std::out_of_range("Second-order reactant species is outside the model");
            if (pair.first > pair.second) std::swap(pair.first, pair.second);
        }

        if (sparse_)
        {
            std::sort(pairs.begin(), pairs.end());
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
            if (pairs.size() > std::numeric_limits<unsigned int>::max())
                throw std::overflow_error("Too many distinct second-order species pairs");
            rows_.assign(size_t(species) + 1, 0);
            columns_.reserve(pairs.size());
            for (const SpeciesPair& pair : pairs)
            {
                ++rows_[size_t(pair.first) + 1];
                columns_.push_back(pair.second);
            }
            for (size_t row = 1; row < rows_.size(); ++row)
                rows_[row] += rows_[row - 1];
            valuesPerSite_ = pairs.size();
        }
        else
            valuesPerSite_ = checkedProduct(species, species);

        const size_t entries = checkedProduct(valuesPerSite_, sites);
        checkedProduct(entries, sizeof(float));
        values_.assign(entries, 0.0f);
    }

    void clear()
    {
        std::fill(values_.begin(), values_.end(), 0.0f);
    }

    // Keep assignment distinct from accumulation. The original integer
    // solver overwrites unlike-species pairs and accumulates self reactions.
    // Accept double so compound assignment rounds only after adding to float.
    void set(unsigned int site, unsigned int a, unsigned int b, double value)
    {
        values_[index(site, a, b)] = value;
        if (!sparse_ && a != b)
            values_[index(site, b, a)] = value;
    }

    void add(unsigned int site, unsigned int a, unsigned int b, double value)
    {
        values_[index(site, a, b)] += value;
        if (!sparse_ && a != b)
            values_[index(site, b, a)] += value;
    }

    bool isSparse() const { return sparse_; }
    size_t valueCount() const { return values_.size(); }
    size_t denseBytes() const { return denseBytes_; }
    size_t storageBytes() const
    {
        return values_.size() * sizeof(float) +
               rows_.size() * sizeof(unsigned int) + columns_.size() * sizeof(unsigned int);
    }
    const std::vector<float>& values() const { return values_; }
    const std::vector<unsigned int>& rows() const { return rows_; }
    const std::vector<unsigned int>& columns() const { return columns_; }

    SecondOrderTableView view() const
    {
        SecondOrderTableView result = {values_.data(), sparse_ ? rows_.data() : NULL,
                                      sparse_ ? columns_.data() : NULL, valuesPerSite_, species_, sites_};
        return result;
    }

private:
    size_t index(unsigned int site, unsigned int a, unsigned int b) const
    {
        if (site >= sites_ || a >= species_ || b >= species_)
            throw std::out_of_range("Second-order propensity index is outside the model");
        if (!sparse_)
            return size_t(site) * valuesPerSite_ + size_t(a) * species_ + b;

        if (a > b) std::swap(a, b);
        const std::vector<unsigned int>::const_iterator begin = columns_.begin() + rows_[a];
        const std::vector<unsigned int>::const_iterator end = columns_.begin() + rows_[size_t(a) + 1];
        const std::vector<unsigned int>::const_iterator entry = std::lower_bound(begin, end, b);
        if (entry == end || *entry != b)
            throw std::out_of_range("Second-order pair was not included in the model topology");
        return size_t(site) * valuesPerSite_ + size_t(entry - columns_.begin());
    }

    unsigned int species_, sites_;
    bool sparse_;
    size_t valuesPerSite_, denseBytes_;
    std::vector<unsigned int> rows_, columns_;
    std::vector<float> values_;
};

}
}

#undef LM_SECOND_ORDER_HD

#endif
