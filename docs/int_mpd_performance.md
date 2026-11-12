# Integer RDME memory and runtime changes

This change targets the single-GPU `IntMpdRdmeSolver` used by
`Minimal_Cell_4DWCM`. It preserves the time step, reaction rates, floating-point
precision, particle-pair accumulation order, reaction-selection order, and
random-hash calls. GPU runtime and peak VRAM must be measured on the target
machine; the results below are allocation formulas, CPU tests, and compilation
checks.

## What changes

### Compact second-order propensities

The previous table allocated one float for every site type and ordered species
pair, including pairs absent from the reaction network. The default now stores
one value for each distinct unordered pair appearing in any second-order
reaction, including self reactions and reactions whose current rate is zero.
A sorted row index finds that pair on the GPU. Missing pairs return zero.

For `S` species, `T` site types and `Q` distinct unordered reactive pairs, the
second-order storage per host/device is:

| Layout | Bytes |
| --- | ---: |
| Previous dense table | `4 * T * S * S` |
| Sparse values and lookup metadata | `4 * T * Q + 4 * Q + 4 * (S + 1)` |

The sparse lookup adds integer operations and binary search. It can improve
memory locality for a sparse network, but its GPU timing benefit depends on
the model and device. A dense network or small model can favor the dense
layout. The zero/first-order tables, reaction-selection data, diffusion table,
and stoichiometric matrix retain their existing representations.

The host computation deliberately preserves legacy semantics: duplicate
unlike-species pair entries use assignment, while self-reaction entries use
accumulation with the existing factor of two. Changing this behavior would be
a separate scientific change.

Use the original layout without rebuilding:

```sh
export LM_INTMPD_SECOND_ORDER_MODE=dense
# Run the normal simulation command.
```

Select the new default explicitly with
`LM_INTMPD_SECOND_ORDER_MODE=sparse`, or unset that variable. A model parameter
named `intMpdSecondOrderMode`, when present, takes precedence over the
environment variable. Accepted values are `sparse` and `dense`.

The solver logs selected layout, allocated table bytes, equivalent dense
bytes, and values per site during setup. Both layouts use the improved
reaction evaluator described below.

### Reaction evaluation without device heap allocation

The shared integer reaction evaluator previously allocated `S` bytes, copied
the entire selected stoichiometric column, consumed reactants, appended
products, and freed the temporary array for every firing site. It now records
consumed reactant IDs in a fixed array bounded by the site's occupancy. It
preserves surviving particle order and appends products in the same ascending
species order, including overflow insertion order within a site.

This removes the per-reaction device heap allocation/free and full temporary
stoichiometric copy. The product scan still visits all species; sparse
stoichiometric products are outside this change. The helper is shared by the
single- and multi-GPU integer solvers. Byte-lattice reaction evaluation is
unchanged.

### Uploads and resource lifetime

After a changed model is successfully copied to the GPU, its dirty flag is
cleared. A failed transfer leaves it dirty for retry. Models with no dynamic
rate changes do not benefit from this particular correction.

Owned model allocations are released at destruction and before their respective
model rebuild. Reaction topology and diffusion-table buffers have separate
cleanup paths, so rebuilding the diffusion model preserves the reaction
topology and permits sparse/dense mode changes. Destruction selects the owning
CUDA device, frees the inherited lattice there, and restores the caller's
previous device. Never-initialized and partially initialized objects are
covered by host tests.

The shared integer and byte diffusion overflow-list size macros now include
the entire four-byte count field. At the default 2,048-overflow capacity,
allocation is 16,388 bytes; the old expression allocated 16,385 bytes.
The per-timestep synchronization and overflow replacement algorithm remain
unchanged.

## Audit an actual model

Once the application has written its finalized `MinCell.lm`, run:

```sh
python3 utils/int_mpd_memory.py /path/to/MinCell.lm
python3 utils/int_mpd_memory.py /path/to/MinCell.lm --json
```

The utility requires `numpy` and `h5py`. It opens the file read-only and reads
reaction dependency columns in blocks. It reports actual species/reaction/site
counts, unique pairs, active second-order species, second-order storage for
both layouts, and the known integer-solver device arrays. Estimates assume
global reaction/stoichiometry/transition matrices, matching the default build.
They exclude CUDA context, overflow buffer, profiling storage, allocator
overhead, and other processes.

For the application's default `64 x 64 x 128` lattice with 16 integer particles
per site, the two particle buffers occupy 64 MiB and the site map another
0.5 MiB. Reaction/model arrays are additional. A simultaneous LAMMPS DNA
subprocess also consumes VRAM, so these estimates alone do not establish that
the whole coupled simulation fits in 16 GB.

## CPU regression tests

Run from this repository with a C++11 compiler on `PATH`:

```sh
python3 -m unittest discover -s testing/host -v
```

There are eight tests. Two use `numpy`/`h5py` and are explicitly skipped if those
packages are unavailable. The remaining tests compile and execute the actual
table helper, extracted host propensity computation, upload branch, cleanup
paths, overflow-size definitions, and shared integer reaction evaluator.
They check:

- Bitwise equality with the dense propensity reference, duplicate/self
  reactions, one-based particle IDs, empty slots, accumulation order, and rate
  updates including zero rates.
- Wide allocation/index arithmetic and a synthetic 100,000-species sparse
  table without allocating its 280 GB dense equivalent.
- One upload per dirty model, failed-upload retry, partial initialization,
  current-device restoration, and sparse-to-dense-to-sparse rebuilds.
- Full overflow-list extent and unchanged survivor/product/overflow order for
  5,000 randomized valid stoichiometries in each global/constant S mode, with
  zero heap allocation inside the actual reaction helper.
- Read-only model auditing against the LM HDF5 schema and rejection of invalid
  second-order dependency counts.

The legacy repository `make check` additionally requires its configured build
environment and `local.mk`. It was attempted in the development workspace but
could not run: the unconfigured Makefile tried linking `TestingHelper.cpp`
without a `main` function. That is separate from the passing host suite above.

## Device compilation without a GPU

An optional smoke check compiles the actual sparse lookup, reaction kernel and
kernel declaration together with the actual shared reaction/random helpers.
It also compiles all distinct restrict-qualified parameter declarations from
the RDME sources and headers, including inactive preprocessor branches.
It requires the NVIDIA NVRTC library and CUDA runtime/compiler headers. A
standard toolkit example is:

```sh
python3 testing/cuda/compile_int_mpd_reaction.py \
  --nvrtc-library /usr/local/cuda/lib64/libnvrtc.so \
  --include-dir /usr/local/cuda/include \
  --ptxas /usr/local/cuda/bin/ptxas \
  --arch 75
```

The script also accepts the corresponding paths from NVIDIA CUDA Python
wheels; multiple `--include-dir` arguments are supported. It compiles both
global-R and constant-R variants with global S. The unused driver-API
`cuda.h` include is omitted from the device-only source because NVRTC does not
provide host libc headers. The actual NVIDIA `cuda_runtime.h` and device
function implementations are used. Temporary PTX/cubin files are discarded.

Validated in the development workspace with NVIDIA NVRTC/ptxas 12.9 for
`compute_75`/`sm_75`: both variants compile and assemble successfully, each
using 72 registers with zero stack-frame bytes and zero spill loads/stores.
This is compiler output, not a runtime benchmark. A full LM CMake/NVCC build,
link, installation, and execution on the user's GPU remain required.

## Manual GPU validation

1. Rebuild and install the changed LM fork using the repository's normal CUDA
   build instructions and the target GPU's architecture. Check that Python
   imports the rebuilt `lm`/`_lm` installation.
2. Start with the same finalized model, initial lattice, fixed nonzero RDME
   seed and replicate ID. Compare a short original build run with the changed
   build in dense mode, then compare changed dense and sparse modes. Keep
   time step, output intervals and all rates identical.
3. Exercise first/second-order and self reactions, zero-rate updates, and an
   overflow-producing model. Use CUDA `compute-sanitizer` to check device
   memory accesses. Existing overflow-list ordering across GPU threads is
   scheduling-dependent, so cross-thread overflow trajectories need careful
   interpretation even when the random-hash stream is unchanged.
4. Record model allocation logs, solver timing, peak process/device memory,
   and host memory for each mode. Then run the coupled whole-cell model long
   enough to include DNA updates; LM and LAMMPS can overlap in GPU residency.
5. Compare output particle counts and lattice contents, and test model rebuild
   and solver destruction in the same Python process. A short run is a
   validation step, not proof of long-run biological equivalence or a fixed
   end-to-end speedup.
