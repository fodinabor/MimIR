#include "mim/plug/ll_pcuda/phase/ll_pcuda.h"

#include <fstream>

#include <mim/driver.h>
#include <mim/plugin.h>

#include <mim/util/sys.h>

#include <mim/plug/gpu/gpu.h>

#include "mim/plug/ll_pcuda/ll_pcuda.h"

using namespace std::string_literals;

namespace mim::plug::ll_pcuda {

namespace {

struct PcudaCompileArgs {
    std::string host_ll_name, dev_ll_name, dev_bc_name;
    bool embed_device_code = true;
};

/// Assembles the textual device IR to the bitcode AdaptiveCpp's HCF images carry.
void compile2bitcode(const PcudaCompileArgs& c) {
    auto llvm_as = sys::require_cmd("llvm-as");
    sys::require_run(std::format("{} {} -o {}", llvm_as, c.dev_ll_name, c.dev_bc_name));
}

std::string slurp_bitcode(const PcudaCompileArgs& c) {
    std::ifstream ifs(c.dev_bc_name, std::ios::binary);
    if (!ifs) fe::throwf("Could not open {} as binary file", c.dev_bc_name);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

} // namespace

class Emit : public Phase {
public:
    Emit(World& world, flags_t annex)
        : Phase(world, annex) {}

    void start() override {
        auto name = world().name() ? std::string(world().name().view()) : "a"s;

        auto c         = PcudaCompileArgs{};
        c.host_ll_name = name + ".ll"s;
        c.dev_ll_name  = name + "_dev.ll"s;
        c.dev_bc_name  = name + "_dev.bc"s;

        auto rt = ll::Emitter::Rt::embed;
        for (const auto& arg : args()) {
            world().DLOG("ll backend arg: `{}`", arg);
            // clang-format off
            if (false) {}
            else if (arg.starts_with("o="))          c.host_ll_name      = arg.substr(2);
            else if (arg.starts_with("output="))     c.host_ll_name      = arg.substr(7);
            else if (arg.starts_with("o-dev="))      c.dev_ll_name       = arg.substr(6);
            else if (arg.starts_with("output-dev=")) c.dev_ll_name       = arg.substr(11);
            else if (arg == "no-embed")              c.embed_device_code = false;
            else if (arg == "rt=embed")              rt                  = ll::Emitter::Rt::embed;
            else if (arg == "rt=extern")             rt                  = ll::Emitter::Rt::ext;
            // clang-format on
        }

        auto split_apply_phase = Phase::create(world().driver().phases(), world().annex<gpu::split_apply>());
        auto setup_phase       = split_apply_phase.get()->expect<RWPhase>("%gpu.split_apply to be an RWPhase");
        setup_phase->run();

        DeviceEmitResult device_result;
        {
            auto dev_ofs  = std::ofstream(c.dev_ll_name);
            device_result = emit_device(setup_phase->new_world(), dev_ofs);
        }

        std::optional<HcfEmbed> hcf;
        if (c.embed_device_code) {
            try {
                compile2bitcode(c);
                auto object_id = hcf_object_id(name, device_result.kernels);
                hcf            = HcfEmbed{
                    .blob
                    = build_hcf(object_id, slurp_bitcode(c), device_result.kernels, device_result.imported_symbols),
                    .object_id = object_id,
                };
            } catch (const sys::CmdNotFound& e) {
                WLOG("{}", e.what());
                WLOG("Falling back to not embedding device code.");
                c.embed_device_code = false;
            }
        }
        auto host_ofs = std::ofstream(c.host_ll_name);
        emit_host(setup_phase->old_world(), host_ofs, hcf, rt);

        if (c.embed_device_code) {
            std::println(std::cout, "Unified LLVM IR with embedded HCF device image written to {}", c.host_ll_name);
        } else {
            std::println(std::cout, "Host-only LLVM IR written to {}", c.host_ll_name);
            std::println(std::cout, "Device-only LLVM IR written to {}", c.dev_ll_name);
        }
    }
};

} // namespace mim::plug::ll_pcuda

using namespace mim;

static void reg_phases(Flags2Phases& phases) { Phase::hook<plug::ll_pcuda::emit, plug::ll_pcuda::Emit>(phases); }

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"ll_pcuda", MIM_VERSION, nullptr, reg_phases}; }
