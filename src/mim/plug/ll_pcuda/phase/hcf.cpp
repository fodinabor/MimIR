#include "mim/plug/ll_pcuda/phase/hcf.h"

// The HCF infrastructure is reused directly from AdaptiveCpp. `hcf_container.hpp` is
// header-only; HIPSYCL_TOOL_COMPONENT keeps its debug channel standalone (no libacpp-rt
// dependency), exactly like AdaptiveCpp's own `acpp-hcf-tool`.
#include <hipSYCL/common/hcf_container.hpp>

namespace mim::plug::ll_pcuda {

namespace {

const char* param_type_name(HcfParamType type) {
    switch (type) {
        case HcfParamType::Integer: return "integer";
        case HcfParamType::FloatingPoint: return "floating-point";
        case HcfParamType::Pointer: return "pointer";
        default: return "other-by-value";
    }
}

} // namespace

uint64_t hcf_object_id(std::string_view world_name, const std::vector<HcfKernel>& kernels) {
    // FNV-1a; deterministic across builds (unlike std::hash) so emitted modules are reproducible.
    uint64_t hash = 0xcbf29ce484222325ull;
    auto mix      = [&hash](std::string_view s) {
        for (unsigned char c : s) {
            hash ^= c;
            hash *= 0x100000001b3ull;
        }
        hash ^= 0xff; // separator so {"ab"} and {"a","b"} differ
        hash *= 0x100000001b3ull;
    };
    mix(world_name);
    for (const auto& kernel : kernels)
        mix(kernel.name);
    return hash;
}

std::string build_hcf(uint64_t object_id,
                      std::string device_bitcode,
                      const std::vector<HcfKernel>& kernels,
                      const std::vector<std::string>& imported_symbols) {
    hipsycl::common::hcf_container hcf;
    auto* root = hcf.root_node();
    root->set("object-id", std::to_string(object_id));
    root->set("generator", "MimIR ll_pcuda backend");

    // AdaptiveCpp's default_llvm_image_selector selects the image by this exact name.
    auto* image = root->add_subnode("images")->add_subnode("llvm-ir.global");
    image->set("variant", "global-module");
    image->set("format", "llvm-ir");
    std::vector<std::string> exported_symbols;
    for (const auto& kernel : kernels)
        exported_symbols.push_back(kernel.name);
    image->set_as_list("exported-symbols", exported_symbols);
    image->set_as_list("imported-symbols", imported_symbols);
    hcf.attach_binary_content(image, device_bitcode);

    auto* kernels_node = root->add_subnode("kernels");
    for (const auto& kernel : kernels) {
        auto* kernel_node = kernels_node->add_subnode(kernel.name);
        kernel_node->set_as_list("image-providers", {"llvm-ir.global"});
        auto* sizes = kernel_node->add_subnode("host-side-parameter-sizes");
        // The runtime's hcf_kernel_info parser expects these subnodes to be present, even if empty.
        kernel_node->add_subnode("compile-flags");
        kernel_node->add_subnode("compile-options");
        auto* params = kernel_node->add_subnode("parameters");
        for (size_t i = 0; i < kernel.params.size(); ++i) {
            sizes->set(std::to_string(i), std::to_string(kernel.params[i].byte_size));
            auto* param = params->add_subnode(std::to_string(i));
            // Free-kernel semantics: each parameter lives in its own host-side `args[i]` slot,
            // so its offset within that slot is always 0 (offsets only accumulate for the
            // SYCL captures-struct convention).
            param->set("byte-offset", "0");
            param->set("byte-size", std::to_string(kernel.params[i].byte_size));
            param->set("original-index", std::to_string(i));
            param->set("type", param_type_name(kernel.params[i].type));
        }
    }

    return hcf.serialize();
}

} // namespace mim::plug::ll_pcuda
