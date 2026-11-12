#!/usr/bin/env python3
"""Compile the actual integer reaction device code without a GPU.

Requires NVIDIA NVRTC plus CUDA runtime/compiler headers. Optionally assembles
the PTX with ptxas. This checks the changed device code and kernel declaration;
it is not a full Lattice Microbes build, link, or execution test.
"""

import argparse
import ctypes as C
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def pointer_parameter_source():
    # Compile actual qualified parameter declarations rather than merely
    # banning a spelling. Include inactive preprocessor branches as well.
    parameters = set()
    count = 0
    for path in sorted((ROOT / "src/rdme").rglob("*")):
        if path.suffix not in (".cu", ".h", ".cuh"):
            continue
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        source = re.sub(r"^\s*#.*$", "", source, flags=re.M)
        declarations = re.findall(r"[^(),;{}]*__restrict__[^(),;{}]*", source)
        qualified_count = source.count("__restrict__")
        if sum(declaration.count("__restrict__") for declaration in declarations) != qualified_count:
            raise RuntimeError("A restrict declaration was not extracted from {}".format(path))
        count += qualified_count
        parameters.update(" ".join(declaration.split()) for declaration in declarations)
    source = '''#include <cuda_runtime.h>
typedef unsigned char uint8_t;
typedef signed char int8_t;
'''
    for index, parameter in enumerate(sorted(parameters)):
        source += "__device__ void pointer_parameter_{}({}) {{}}\n".format(index, parameter)
    print("Checking {} restrict declarations ({} distinct parameters)".format(count, len(parameters)), flush=True)
    return source


def device_source():
    header = (ROOT / "src/rdme/SecondOrderPropensityTable.h").read_text()
    view = header[header.index("template <typename T>"):header.index("// The topology")]
    constants = (ROOT / "src/cuda/constant.cu").read_text()
    constants = constants[constants.index("__constant__ unsigned int gpuidC;"):]
    declaration = (ROOT / "src/rdme/IntMpdRdmeSolver.h").read_text().split("namespace intmpdrdme_dev {", 1)[1]
    declaration = declaration.split("#ifdef MPD_CUDA_3D_GRID_LAUNCH", 1)[0]
    source = (ROOT / "src/rdme/IntMpdRdmeSolver.cu").read_text()
    kernel = source[source.rindex("#ifdef MPD_GLOBAL_R_MATRIX\n__global__ void __launch_bounds__"):]
    kernel = kernel[:kernel.rindex("\n}\n}\n\n}\n}") + 2]
    return ('''#include <cuda_runtime.h>
#include "config.h"
typedef unsigned int uint;
typedef unsigned char uint8_t;
typedef signed char int8_t;
#define LM_SECOND_ORDER_HD __host__ __device__
''' + view + constants + declaration + '''
#include "rdme/dev/xor_random_dev.cu"
#include "rdme/dev/word_reaction_dev.cu"
''' + kernel)


def compile_device(args):
    library = C.CDLL(str(args.nvrtc_library))
    library.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                         C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)]
    library.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    for name in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
        getattr(library, name).argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    for name in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
        getattr(library, name).argtypes = [C.c_void_p, C.c_char_p]
    library.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]

    kernel_source = device_source()
    config = '''#define MPD_GLOBAL_S_MATRIX
#define MPD_GLOBAL_T_MATRIX
#define MPD_FREAKYFAST
#define MPD_WORDS_PER_SITE 16
#define MPD_MAX_REACTION_TABLE_ENTRIES 32
#define TUNE_MPD_MAX_PARTICLE_OVERFLOWS 2048
#define TUNE_MPD_REACTION_BLOCK_X_SIZE 32
#define TUNE_MPD_REACTION_BLOCK_Y_SIZE 8
#define RNG_CUDA_DOUBLE_PRECISION
'''
    with tempfile.TemporaryDirectory() as directory:
        compilations = [("pointer_declarations", pointer_parameter_source(), config),
                        ("global_r", kernel_source, config + "#define MPD_GLOBAL_R_MATRIX\n"),
                        ("constant_r", kernel_source, config)]
        for mode, source, model_config in compilations:
            headers = {"config.h": model_config}
            for name in ("rdme/dev/xor_random_dev.cu", "rdme/dev/word_reaction_dev.cu"):
                # Neither device helper calls the driver API. Its unused
                # cuda.h include needs host libc headers unavailable to NVRTC;
                # cuda_runtime.h remains the real NVIDIA runtime header.
                headers[name] = (ROOT / "src" / name).read_text().replace(
                    "#include <cuda.h>", "// Unused driver API include omitted for device-only NVRTC.")
            program = C.c_void_p()
            status = library.nvrtcCreateProgram(
                C.byref(program), source.encode(), b"int_mpd_reaction.cu", len(headers),
                (C.c_char_p * len(headers))(*(value.encode() for value in headers.values())),
                (C.c_char_p * len(headers))(*(name.encode() for name in headers)))
            if status:
                raise RuntimeError("nvrtcCreateProgram failed: {}".format(status))
            try:
                options = ["--gpu-architecture=compute_" + args.arch, "--std=c++11", "-I" + str(ROOT / "src")]
                options.extend("-I" + str(path) for path in args.include_dir)
                status = library.nvrtcCompileProgram(program, len(options),
                                                     (C.c_char_p * len(options))(*(option.encode() for option in options)))
                size = C.c_size_t()
                library.nvrtcGetProgramLogSize(program, C.byref(size))
                log = C.create_string_buffer(size.value)
                library.nvrtcGetProgramLog(program, log)
                if log.value:
                    print(log.value.decode())
                if status:
                    raise RuntimeError("NVRTC {} compilation failed: {}".format(mode, status))
                library.nvrtcGetPTXSize(program, C.byref(size))
                ptx = C.create_string_buffer(size.value)
                library.nvrtcGetPTX(program, ptx)
                print("PASS: {} compiled for compute_{} ({} PTX bytes)".format(mode, args.arch, size.value), flush=True)
                if args.ptxas and mode != "pointer_declarations":
                    filename = Path(directory) / (mode + ".ptx")
                    filename.write_bytes(ptx.value)
                    subprocess.run([str(args.ptxas), "-arch=sm_" + args.arch, str(filename),
                                    "-o", str(filename.with_suffix(".cubin")), "-v"], check=True)
            finally:
                library.nvrtcDestroyProgram(C.byref(program))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvrtc-library", type=Path, required=True)
    parser.add_argument("--include-dir", type=Path, action="append", required=True)
    parser.add_argument("--arch", default="75", help="CUDA architecture number, e.g. 75 or 86")
    parser.add_argument("--ptxas", type=Path, help="Optional PTX assembler binary")
    compile_device(parser.parse_args())
