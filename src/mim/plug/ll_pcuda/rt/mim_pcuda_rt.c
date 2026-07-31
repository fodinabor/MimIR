// Runtime wrappers for the MimIR `ll_pcuda` backend.
//
// Like `ll`'s `mim_rt.c`, this file is compiled to textual LLVM IR by `clang` at build time (see
// `add_mim_runtime` in `cmake/Mim.cmake`) and embedded into or linked with the host module emitted
// by the `ll_pcuda` backend (`-X ll_pcuda:rt=embed|extern`); see issue #486.
//
// It intentionally does not include AdaptiveCpp's pCUDA headers: the wrappers only touch the
// `pcudaError_t` return codes that the backend already has in hand, so they stay self-contained
// and build without AdaptiveCpp. The host module links against `libacpp-rt.so` anyway, which
// provides `pcudaGetErrorName`/`pcudaGetErrorString` for the diagnostics below.

#include <stdio.h>
#include <stdlib.h>

extern const char* pcudaGetErrorName(int error);
extern const char* pcudaGetErrorString(int error);

/// Aborts the program if a pCUDA runtime call returned a non-zero `pcudaError_t`.
/// The backend emits a call to this after every pCUDA runtime call instead of open-coding error handling.
void mim_pcuda_check(int result) {
    if (result != 0) {
        fprintf(stderr, "MimIR: pCUDA error: %s: %s\n", pcudaGetErrorName(result), pcudaGetErrorString(result));
        abort();
    }
}
