#pragma once

#include <cstdint>

#include <string>
#include <string_view>
#include <vector>

namespace mim::plug::ll_pcuda {

/// Parameter categories understood by AdaptiveCpp's `hcf_kernel_info` parser
/// (see `hipSYCL/runtime/kernel_cache.hpp`); only Pointer is treated specially there.
enum class HcfParamType { Integer, FloatingPoint, Pointer, OtherByValue };

struct HcfParam {
    size_t byte_size;
    HcfParamType type;
};

/// Metadata of one kernel in the device image; HCF kernel node name, LLVM function name in the
/// device module, and the string passed to `__pcudaKernelCall` must all be HcfKernel::name.
struct HcfKernel {
    std::string name;
    std::vector<HcfParam> params; ///< Flat "free kernel" parameters, in order.
};

/// Computes a stable id for the HCF object (FNV-1a over the world and kernel names).
/// AdaptiveCpp's `hcf_cache` keys registered HCF objects by this id, so it has to be unique
/// per device image within the final executable.
uint64_t hcf_object_id(std::string_view world_name, const std::vector<HcfKernel>& kernels);

/// Serializes the kernel metadata and the device-LLVM-bitcode into AdaptiveCpp's Heterogeneous
/// Container Format, mirroring what AdaptiveCpp's SSCP `TargetSeparationPass` emits (`generateHCF`).
/// The result is embedded into the host module and registered with `__acpp_register_hcf`.
std::string build_hcf(uint64_t object_id,
                      std::string device_bitcode,
                      const std::vector<HcfKernel>& kernels,
                      const std::vector<std::string>& imported_symbols);

} // namespace mim::plug::ll_pcuda
