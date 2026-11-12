#!/usr/bin/env python3
"""Read a finalized LM model and estimate IntMpd's device-array allocations.

This does not run the simulation or open CUDA. It reads dependency columns in
blocks so auditing a large model does not materialize its entire dense matrix.
Requires the numpy and h5py packages normally used by jLM.
"""

import argparse
import json


def audit_model(filename):
    import h5py
    import numpy as np

    with h5py.File(filename, "r") as file:
        diffusion = file["Model/Diffusion"]
        reactions = file["Model/Reaction"]
        species = int(diffusion.attrs["numberSpecies"])
        reaction_count = int(diffusion.attrs["numberReactions"])
        site_types = int(diffusion.attrs["numberSiteTypes"])
        dimensions = [int(diffusion.attrs["lattice" + axis + "Size"]) for axis in "XYZ"]
        occupancy = int(diffusion.attrs["particlesPerSite"])
        bytes_per_particle = int(diffusion.attrs.get("bytes_per_particle", 1))
        if bytes_per_particle != 4:
            raise ValueError("IntMpd requires an integer lattice with bytes_per_particle=4")
        if int(reactions.attrs["numberSpecies"]) != species or int(reactions.attrs.get("numberReactions", 0)) != reaction_count:
            raise ValueError("Reaction and diffusion model dimensions disagree")

        pairs = set()
        if reaction_count:
            dependencies = reactions["DependencyMatrix"]
            types = reactions["ReactionTypes"]
            if dependencies.shape != (species, reaction_count) or types.shape != (reaction_count,):
                raise ValueError("ReactionTypes or DependencyMatrix has unexpected dimensions")
            for begin in range(0, reaction_count, 128):
                end = min(begin + 128, reaction_count)
                block_types = types[begin:end]
                if np.any((block_types < 0) | (block_types > 3)):
                    raise ValueError("The integer RDME solver supports reaction types 0 through 3")
                if not np.any((block_types == 2) | (block_types == 3)):
                    continue
                block = dependencies[:, begin:end]
                for offset, kind in enumerate(block_types):
                    if kind not in (2, 3):
                        continue
                    reactants = np.flatnonzero(block[:, offset] == 1)
                    expected = 2 if kind == 2 else 1
                    if len(reactants) != expected:
                        raise ValueError("Invalid dependency count for reaction {}: expected {}, found {}".format(
                            begin + offset, expected, len(reactants)))
                    a = int(reactants[0])
                    b = int(reactants[-1])
                    pairs.add((a, b))

    pair_count = len(pairs)
    dense = 4 * species * species * site_types
    sparse = 4 * pair_count * site_types + 4 * (species + 1) + 4 * pair_count
    sites = dimensions[0] * dimensions[1] * dimensions[2]
    arrays = {
        "particle_lattices": 2 * sites * occupancy * bytes_per_particle,
        "site_lattice": sites,
        "stoichiometric_matrix": species * reaction_count,
        "reaction_locations": reaction_count * site_types,
        "transition_probabilities": 4 * species * site_types * site_types,
        "reaction_parameters": 20 * reaction_count,
        "zero_order_propensities": 4 * site_types,
        "first_order_propensities": 4 * species * site_types,
    }
    return {
        "species": species,
        "reactions": reaction_count,
        "site_types": site_types,
        "lattice_dimensions": dimensions,
        "particles_per_site": occupancy,
        "unique_second_order_pairs": pair_count,
        "active_second_order_species": len({species_id for pair in pairs for species_id in pair}),
        "second_order_dense_bytes": dense,
        "second_order_sparse_bytes": sparse,
        "second_order_saved_bytes": dense - sparse,
        "device_arrays_bytes": arrays,
        "estimated_device_arrays_dense_bytes": sum(arrays.values()) + dense,
        "estimated_device_arrays_sparse_bytes": sum(arrays.values()) + sparse,
        "scope": "Integer lattice; global R/S/T matrices; excludes CUDA context, overflow buffer, profiler, other solvers, and allocator overhead. Estimates are not measured VRAM.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Finalized MinCell.lm (read only)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args()
    try:
        report = audit_model(args.model)
    except (ImportError, OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print("Species: {species}; reactions: {reactions}; site types: {site_types}".format(**report))
    print("Unique second-order pairs: {unique_second_order_pairs}; active reactant species: {active_second_order_species}".format(**report))
    for label, key in (("Dense second-order table", "second_order_dense_bytes"),
                       ("Sparse second-order table", "second_order_sparse_bytes"),
                       ("Second-order bytes saved", "second_order_saved_bytes"),
                       ("Known device arrays, dense", "estimated_device_arrays_dense_bytes"),
                       ("Known device arrays, sparse", "estimated_device_arrays_sparse_bytes")):
        value = report[key]
        print("{}: {:,} bytes ({:.3f} MiB)".format(label, value, value / 2**20))
    print(report["scope"])


if __name__ == "__main__":
    main()
