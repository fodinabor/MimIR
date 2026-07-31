#include "mim/plug/ll_pcuda/phase/ll_pcuda.h"

#include <format>
#include <iomanip>
#include <sstream>

#include <mim/driver.h>

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/ll_pcuda/ll_pcuda.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

using namespace std::string_literals;

namespace mim::plug::ll_pcuda {

namespace core = mim::plug::core;
namespace gpu  = mim::plug::gpu;
namespace ll   = mim::plug::ll;
namespace math = mim::plug::math;
namespace mem  = mim::plug::mem;

class HostEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    HostEmitter(World& world, std::ostream& ostream, const std::optional<HcfEmbed>& hcf)
        : Super(world, "llvm_pcuda_host_emitter", ostream)
        , hcf_(hcf) {}

    void start() final;
    void find_kernels(const Def*);

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

protected:
    std::string convert(const Def*, bool simd = true) override;

private:
    static constexpr std::string_view hcf_content_name_     = "@__acpp_local_sscp_hcf_content";
    static constexpr std::string_view hcf_object_id_name_   = "@__acpp_local_sscp_hcf_object_id";
    static constexpr std::string_view kernel_name_prefix    = "@.kname.";
    static constexpr std::string_view kernel_storage_prefix = "@.kstorage.";

    void emit_pcuda_error_handling(ll::BB&, const std::string&, bool at_tail = false);
    void emit_hcf_embedding();

    const std::optional<HcfEmbed>& hcf_;
    LamMap<int> kernel_ids_;

    DefSet analyzed_;
};

class DeviceEmitter : public ll::Emitter {
public:
    using Super = ll::Emitter;

    DeviceEmitter(World& world, std::ostream& ostream)
        : Super(world, "llvm_pcuda_device_emitter", ostream) {}

    void start() final;

    std::string prepare() override;

    std::optional<std::string> isa_targetspecific_intrinsic(ll::BB&, const Def*) final;

    DeviceEmitResult& result() { return result_; }

private:
    /// Canonical SSCP device IR uses opaque `ptr` for all pointer types — no explicit address
    /// space. The JIT's `llvm-to-backend` passes assign the right address spaces during backend
    /// flavoring; an explicit addrspace(N) in the stage-1 IR breaks e.g. the AMDGPU lowering.
    std::string convert(const Def* def, bool simd = false) override {
        if (simd) WLOG("Ignoring simd=true for type conversion in device code.");
        if (Axm::isa<mem::Ptr>(def)) return "ptr";
        return Super::convert(def, false);
    }

    /// Device slots live in a module-scope global in their requested address space, not on the
    /// stack; the addrspacecast yields the generic pointer the rest of the body works with.
    std::string emit_slot(ll::BB& bb, const App* app, const Def* pointee, const Def* addr_space) override {
        auto addr_space_lit = Lit::isa(addr_space);
        if (addr_space_lit.value_or(0) == 0) return Super::emit_slot(bb, app, pointee, addr_space);
        auto v_ptr = "@" + app->unique_name() + ".slot";
        std::print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", v_ptr, *addr_space_lit,
                   convert(pointee));
        return std::format("addrspacecast (ptr addrspace({}) {} to ptr)", *addr_space_lit, v_ptr);
    }

    /// Like declare(), but also records the builtin as an imported symbol of the device image.
    template<class... Args>
    void declare_sscp(std::string_view name, std::format_string<Args...> s, Args&&... args) {
        if (std::ranges::find(result_.imported_symbols, name) == result_.imported_symbols.end())
            result_.imported_symbols.emplace_back(name);
        declare(s, std::forward<Args>(args)...);
    }

    LamSet kernels_;
    DeviceEmitResult result_;
};

namespace {

/// Builds an LLVM byte-string literal `c"\XX\XX..."` for arbitrary bytes.
std::string llvm_byte_literal(std::string_view bytes) {
    std::ostringstream s;
    s << "c\"" << std::hex << std::setfill('0');
    for (unsigned char c : bytes)
        s << "\\" << std::setw(2) << static_cast<int>(c);
    s << "\"";
    return s.str();
}

/// Builds a NUL-terminated LLVM C-string literal for a kernel name.
std::string llvm_cstring_literal(std::string_view s) {
    std::ostringstream out;
    out << "c\"" << std::hex << std::setfill('0');
    for (unsigned char c : s)
        if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\')
            out << static_cast<char>(c);
        else
            out << "\\" << std::setw(2) << static_cast<int>(c);
    out << "\\00\"";
    return out.str();
}

} // namespace

void HostEmitter::start() {
    for (auto def : world().annexes().defs())
        find_kernels(def);
    for (auto def : world().externals().muts())
        find_kernels(def);

    emit_hcf_embedding();

    Super::start();
}

void HostEmitter::find_kernels(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    for (auto d : def->deps())
        find_kernels(d);

    if (auto launch = Axm::isa<gpu::launch>(def)) {
        auto kernel     = launch->decurry()->decurry()->arg();
        auto kernel_lam = kernel->expect_mut<Lam>("the kernel passed to %gpu.launch to be a mutable lambda");
        if (kernel_ids_.contains(kernel_lam)) return;
        auto kid                = kernel_ids_.size();
        kernel_ids_[kernel_lam] = kid;
    }
}

void HostEmitter::emit_hcf_embedding() {
    // Stage-1 IR constants; same layout AdaptiveCpp's SSCP TargetSeparationPass leaves behind
    // after initialization, so the runtime contract is identical.
    if (hcf_.has_value()) {
        auto n = hcf_->blob.size();
        std::print(vars_decls_, "{} = private constant [{} x i8] {}\n", hcf_content_name_, n,
                   llvm_byte_literal(hcf_->blob));
        // LLVM's IR parser wants i64 literals in signed representation; the bit pattern is what
        // the runtime compares against the HCF's (unsigned) object-id string.
        std::print(vars_decls_, "{} = private constant i64 {}\n", hcf_object_id_name_,
                   static_cast<int64_t>(hcf_->object_id));

        // Registers the HCF blob with AdaptiveCpp's kernel cache at program startup — the whole
        // contract between the emitted host module and libacpp-rt.so's HCF infrastructure.
        declare("void @__acpp_register_hcf(ptr, i64)");
        std::print(func_impls_,
                   "define internal void @.mimir_register_hcf_ctor() {{\n"
                   "  call void @__acpp_register_hcf(ptr {}, i64 {})\n"
                   "  ret void\n"
                   "}}\n\n",
                   hcf_content_name_, n);
        std::print(vars_decls_, "@llvm.global_ctors = appending global [1 x {{ i32, ptr, ptr }}] "
                                "[{{ i32, ptr, ptr }} {{ i32 65535, ptr @.mimir_register_hcf_ctor, ptr null }}]\n");
    } else {
        std::print(vars_decls_, "; Add the bytes of your serialized HCF device image here (and register it\n"
                                "; with __acpp_register_hcf in a global constructor):\n");
        std::print(vars_decls_, "{} = private constant [YOUR_HCF_SIZE_GOES_HERE x i8] YOUR_HCF_DATA_GOES_HERE\n",
                   hcf_content_name_);
        std::print(vars_decls_, "{} = private constant i64 YOUR_HCF_OBJECT_ID_GOES_HERE\n", hcf_object_id_name_);
    }

    // Per-kernel name string and kernel_specific_storage slot — __pcudaKernelCall identifies a
    // kernel by (object-id, name) and memoizes its lookup in the storage slot.
    for (auto [kernel, kid] : kernel_ids_) {
        auto name = id(kernel).substr(1);
        std::print(vars_decls_, "{}{} = private constant [{} x i8] {}\n", kernel_name_prefix, kid, name.size() + 1,
                   llvm_cstring_literal(name));
        std::print(vars_decls_, "{}{} = internal global ptr null\n", kernel_storage_prefix, kid);
    }
}

constexpr auto Pcuda_Set_Device              = "pcudaSetDevice";
constexpr auto Pcuda_Malloc                  = "pcudaAllocateDevice";
constexpr auto Pcuda_Free                    = "pcudaFree";
constexpr auto Pcuda_Memcpy                  = "pcudaMemcpy";
constexpr auto Pcuda_Memcpy_Async            = "pcudaMemcpyAsync";
constexpr auto Pcuda_Stream_Create           = "pcudaStreamCreate";
constexpr auto Pcuda_Stream_Destroy          = "pcudaStreamDestroy";
constexpr auto Pcuda_Stream_Sync             = "pcudaStreamSynchronize";
constexpr auto Pcuda_Push_Call_Configuration = "__pcudaPushCallConfiguration";
constexpr auto Pcuda_Kernel_Call             = "__pcudaKernelCall";

// pcudaMemcpyKind (mirrors cudaMemcpyKind)
constexpr auto Pcuda_Memcpy_Htod = 1;
constexpr auto Pcuda_Memcpy_Dtoh = 2;

void HostEmitter::emit_pcuda_error_handling(ll::BB& bb, const std::string& pcuda_result, bool at_tail) {
    // Offload the pcudaError_t check to the C runtime wrapper `mim_pcuda_check`
    // (see rt/mim_pcuda_rt.c), like `ll_nvptx`'s `mim_cu_check`.
    declare_rt("void @mim_pcuda_check(i32)");
    if (at_tail)
        bb.tail("call void @mim_pcuda_check(i32 {})", pcuda_result);
    else
        std::print(bb.body().emplace_back(), "call void @mim_pcuda_check(i32 {})", pcuda_result);
}

std::string HostEmitter::convert(const Def* type, bool simd) {
    if (auto ptr = Axm::isa<mem::Ptr>(type)) {
        auto [_, addr_space] = ptr->args<2>();
        auto lit             = Lit::isa(addr_space);
        if (lit.value_or(0L) != 0) {
            // Device pointers stay opaque `ptr`s in host code — the pCUDA runtime ABI takes them
            // as plain pointers (unlike the CUDA driver API's i64 device pointers).
            return "ptr";
        }
    }
    return Super::convert(type, simd);
}

std::optional<std::string> HostEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    if (auto default_stream = Axm::isa<gpu::default_stream>(def)) {
        // The null stream is the pCUDA runtime's default in-order queue.
        return "null";
    } else if (auto init = Axm::isa<gpu::init>(def)) {
        // The pCUDA runtime initializes lazily; selecting device 0 both forces initialization
        // and fails early (via mim_pcuda_check) when no device is available. There is no module
        // to load — the HCF image was registered by the global constructor already.
        declare("i32 @{}(i32)", Pcuda_Set_Device);
        auto init_res = bb.assign(name + "_setdev_res", "call i32 @{}(i32 0)", Pcuda_Set_Device);
        emit_pcuda_error_handling(bb, init_res);

        auto mem = init->arg();
        return emit_unsafe(mem);
    } else if (auto deinit = Axm::isa<gpu::deinit>(def)) {
        // Nothing to tear down: streams are destroyed via %gpu.stream_deinit and the runtime
        // cleans up its queues and the registered HCF objects at program exit.
        emit_unsafe(deinit->arg(0));
        return emit_unsafe(deinit->arg(1));
    } else if (auto stream_init = Axm::isa<gpu::stream_init>(def)) {
        declare("i32 @{}(ptr)", Pcuda_Stream_Create);

        emit_unsafe(stream_init->arg(0));
        emit_unsafe(stream_init->arg(1));
        auto stream_ptr = emit(stream_init->arg(2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", Pcuda_Stream_Create, stream_ptr);
        emit_pcuda_error_handling(bb, res);
        return res;
    } else if (auto stream_deinit = Axm::isa<gpu::stream_deinit>(def)) {
        declare("i32 @{}(ptr)", Pcuda_Stream_Destroy);

        emit_unsafe(stream_deinit->arg(0));
        emit_unsafe(stream_deinit->arg(1));
        auto stream = emit(stream_deinit->arg(2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", Pcuda_Stream_Destroy, stream);
        emit_pcuda_error_handling(bb, res);
        return res;
    } else if (auto stream_sync = Axm::isa<gpu::stream_sync>(def)) {
        declare("i32 @{}(ptr)", Pcuda_Stream_Sync);

        emit_unsafe(stream_sync->arg(0));
        emit_unsafe(stream_sync->arg(1));
        auto stream = emit(stream_sync->arg(2));

        auto res = bb.assign(name, "call i32 @{}(ptr {})", Pcuda_Stream_Sync, stream);
        emit_pcuda_error_handling(bb, res);
        return res;
    } else if (auto alloc = Axm::isa<gpu::alloc>(def)) {
        bool is_async;
        switch (alloc.id()) {
            case gpu::alloc::block: is_async = false; break;
            case gpu::alloc::asyn: is_async = true; break;
            default: fe::throwf("ll_pcuda backend: unhandled %gpu.alloc id in '{}'", def);
        }

        // The pCUDA runtime has no asynchronous allocation entry point (pcudaMallocAsync is a
        // header-only template forwarding to pcudaAllocateDevice), so both flavors block.
        declare("i32 @{}(ptr, i64)", Pcuda_Malloc);

        emit_unsafe(alloc->arg(0));
        if (is_async) emit_unsafe(alloc->arg(1));
        auto alloc_t    = alloc->decurry()->arg();
        World& w        = alloc_t->world();
        auto type_size  = w.call(core::trait::size, alloc_t);
        auto alloc_size = emit(type_size);

        auto ptr_t = convert(Axm::expect<mem::Ptr>(def->proj(1)->type(), "a %mem.Ptr"));

        auto alloc_ptr = bb.assign(name + "ptr", "alloca {}", ptr_t);
        auto alloc_res = bb.assign(name + "res", "call i32 @{}(ptr {}, i64 {})", Pcuda_Malloc, alloc_ptr, alloc_size);
        emit_pcuda_error_handling(bb, alloc_res);
        return bb.assign(name, "load {}, ptr {}", ptr_t, alloc_ptr);
    } else if (auto free = Axm::isa<gpu::free>(def)) {
        bool is_async;
        switch (free.id()) {
            case gpu::free::block: is_async = false; break;
            case gpu::free::asyn: is_async = true; break;
            default: fe::throwf("ll_pcuda backend: unhandled %gpu.free id in '{}'", def);
        }

        // Like allocation, deallocation only exists in a blocking flavor.
        declare("i32 @{}(ptr)", Pcuda_Free);

        emit_unsafe(free->arg(0));
        auto ptr = emit(free->arg(1));
        if (is_async) emit_unsafe(free->arg(2));

        auto free_res = bb.assign(name + "res", "call i32 @{}(ptr {})", Pcuda_Free, ptr);
        emit_pcuda_error_handling(bb, free_res);
        return free_res;
    } else if (auto copy_to_device = Axm::isa<gpu::copy_to_device>(def)) {
        bool is_async;
        switch (copy_to_device.id()) {
            case gpu::copy_to_device::block: is_async = false; break;
            case gpu::copy_to_device::asyn: is_async = true; break;
            default: fe::throwf("ll_pcuda backend: unhandled %gpu.copy_to_device id in '{}'", def);
        }

        if (is_async)
            declare("i32 @{}(ptr, ptr, i64, i32, ptr)", Pcuda_Memcpy_Async);
        else
            declare("i32 @{}(ptr, ptr, i64, i32)", Pcuda_Memcpy);

        auto type      = copy_to_device->decurry()->arg();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_device->arg(0));
        emit_unsafe(copy_to_device->arg(1));
        auto host_ptr = emit(copy_to_device->arg(2));
        auto dev_ptr  = emit(copy_to_device->arg(3));
        auto size     = emit(type_size);

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_device->arg(4));
            copy_res    = bb.assign(name + "res", "call i32 @{}(ptr {}, ptr {}, i64 {}, i32 {}, ptr {})",
                                    Pcuda_Memcpy_Async, dev_ptr, host_ptr, size, Pcuda_Memcpy_Htod, stream);
        } else
            copy_res = bb.assign(name + "res", "call i32 @{}(ptr {}, ptr {}, i64 {}, i32 {})", Pcuda_Memcpy, dev_ptr,
                                 host_ptr, size, Pcuda_Memcpy_Htod);

        emit_pcuda_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto copy_to_host = Axm::isa<gpu::copy_to_host>(def)) {
        bool is_async;
        switch (copy_to_host.id()) {
            case gpu::copy_to_host::block: is_async = false; break;
            case gpu::copy_to_host::asyn: is_async = true; break;
            default: fe::throwf("ll_pcuda backend: unhandled %gpu.copy_to_host id in '{}'", def);
        }

        if (is_async)
            declare("i32 @{}(ptr, ptr, i64, i32, ptr)", Pcuda_Memcpy_Async);
        else
            declare("i32 @{}(ptr, ptr, i64, i32)", Pcuda_Memcpy);

        auto [type]    = copy_to_host->decurry()->args<1>();
        World& w       = type->world();
        auto type_size = w.call(core::trait::size, type);

        emit_unsafe(copy_to_host->arg(0));
        emit_unsafe(copy_to_host->arg(1));
        auto dev_ptr  = emit(copy_to_host->arg(2));
        auto host_ptr = emit(copy_to_host->arg(3));
        auto size     = emit(type_size);

        std::string copy_res;
        if (is_async) {
            auto stream = emit(copy_to_host->arg(4));
            copy_res    = bb.assign(name + "res", "call i32 @{}(ptr {}, ptr {}, i64 {}, i32 {}, ptr {})",
                                    Pcuda_Memcpy_Async, host_ptr, dev_ptr, size, Pcuda_Memcpy_Dtoh, stream);
        } else
            copy_res = bb.assign(name + "res", "call i32 @{}(ptr {}, ptr {}, i64 {}, i32 {})", Pcuda_Memcpy, host_ptr,
                                 dev_ptr, size, Pcuda_Memcpy_Dtoh);

        emit_pcuda_error_handling(bb, copy_res);
        return copy_res;
    } else if (auto launch = Axm::isa<gpu::launch>(def)) {
        declare("void @{}(i64, i32, i64, i32, i64, ptr)", Pcuda_Push_Call_Configuration);
        declare("i32 @{}(ptr, ptr, i64, ptr)", Pcuda_Kernel_Call);

        auto [implicits, launch_config, kernel_def, arg_def, func_args] = launch->uncurry_args<5>();
        auto [n_groups_def, n_items_def, stream_def, m, MT]             = launch_config->projs<5>();
        auto [mem, ret_lam_def]                                         = func_args->projs<2>();

        Lam* lam = kernel_def->isa_mut<Lam>();
        if (!lam) fe::throwf("kernel is not a lamda {}", kernel_def);
        if (!kernel_ids_.contains(lam)) fe::throwf("unknown kernel {}", lam);
        auto kid = kernel_ids_[lam];

        uint64_t shared_mem_bytes = 0;
        if (auto smem_count = Lit::expect(m, "a shared-memory allocation count")) {
            if (smem_count != 1) fe::throwf("You can only have one dynamic allocation of shared memory per kernel");
            shared_mem_bytes = Lit::expect(world().call(core::trait::size, MT), "a shared-memory size");
        }

        emit_unsafe(mem);
        auto n_groups = emit(n_groups_def);
        auto n_items  = emit(n_items_def);
        auto stream   = emit(stream_def);
        auto ret_lam  = emit(ret_lam_def);

        // Pack each dim3 as the x86-64 SysV coercion clang produces for libacpp-rt:
        // dim3{x, y = 1, z = 1} becomes (i64 xy = (1 << 32) | x, i32 z = 1).
        constexpr uint64_t y_eq_one_high = uint64_t(1) << 32;
        auto grid_x64                    = bb.assign(name + "_grid_x64", "zext i32 {} to i64", n_groups);
        auto grid_xy                     = bb.assign(name + "_grid_xy", "or i64 {}, {}", grid_x64, y_eq_one_high);
        auto blk_x64                     = bb.assign(name + "_blk_x64", "zext i32 {} to i64", n_items);
        auto blk_xy                      = bb.assign(name + "_blk_xy", "or i64 {}, {}", blk_x64, y_eq_one_high);

        std::print(bb.body().emplace_back(), "call void @{}(i64 {}, i32 1, i64 {}, i32 1, i64 {}, ptr {})",
                   Pcuda_Push_Call_Configuration, grid_xy, blk_xy, shared_mem_bytes, stream);

        // Free-kernel argument passing: one host-side slot per flat kernel argument, packed into
        // an array of pointers to the slots (matching the HCF host-side-parameter-sizes).
        auto arg_arity = arg_def->num_projs();
        auto args_arr  = bb.assign(name + "_args", "alloca [{} x ptr]", arg_arity);
        for (size_t i = 0; i < arg_arity; ++i) {
            const Def* proj = arg_arity > 1 ? arg_def->proj(i) : arg_def;
            auto proj_val   = emit(proj);
            auto proj_type  = convert(proj->type());
            auto slot       = bb.assign(std::format("{}_arg{}_slot", name, i), "alloca {}", proj_type);
            std::print(bb.body().emplace_back(), "store {} {}, ptr {}", proj_type, proj_val, slot);
            auto gep = bb.assign(std::format("{}_arg{}_gep", name, i),
                                 "getelementptr [{} x ptr], ptr {}, i64 0, i64 {}", arg_arity, args_arr, i);
            std::print(bb.body().emplace_back(), "store ptr {}, ptr {}", slot, gep);
        }

        auto object_id = bb.assign(name + "_objid", "load i64, ptr {}", hcf_object_id_name_);
        auto launch_res
            = bb.assign(name + "_kcall", "call i32 @{}(ptr {}{}, ptr {}, i64 {}, ptr {}{})", Pcuda_Kernel_Call,
                        kernel_name_prefix, kid, args_arr, object_id, kernel_storage_prefix, kid);
        emit_pcuda_error_handling(bb, launch_res);
        return ret_lam;
    }
    return std::nullopt;
}

void DeviceEmitter::start() {
    for (auto kernel : world().externals().muts()) {
        auto kernel_lam = kernel->expect_mut<Lam>("an external kernel to be a mutable lambda");
        kernels_.emplace(kernel_lam);

        // Collect the HCF kernel metadata: the user-data argument (see prepare() for the kernel
        // var layout) becomes the flat "free kernel" parameter list.
        auto& hcf_kernel = result_.kernels.emplace_back();
        hcf_kernel.name  = id(kernel_lam).substr(1);
        auto arg         = kernel_lam->var(7);
        auto arity       = arg->num_projs();
        for (size_t i = 0; i < arity; ++i) {
            auto type = (arity > 1 ? arg->proj(i) : arg)->type();
            if (Axm::isa<mem::Ptr>(type)) {
                hcf_kernel.params.emplace_back(HcfParam{8, HcfParamType::Pointer});
                continue;
            }
            auto size = Lit::expect(world().call(core::trait::size, type), "a kernel-parameter size");
            auto kind = Idx::isa_lit(type) ? HcfParamType::Integer
                      : math::isa_f(type)  ? HcfParamType::FloatingPoint
                                           : HcfParamType::OtherByValue;
            hcf_kernel.params.emplace_back(HcfParam{size, kind});
        }
    }

    Super::start();

    // SSCP kernel-discovery metadata: !hipsycl.sscp.annotations names each kernel function and
    // its dimensionality; without it the CPU (CBS) lowering cannot find the kernels. MimIR's GPU
    // model has a single group_id/item_id per launch, so the dimension is always 1.
    std::ostringstream md;
    int next_id = 0;
    if (!kernels_.empty()) {
        for (auto kernel : kernels_)
            std::print(md, "!{} = !{{ptr {}, !\"hipsycl_kernel_dimension\", i32 1}}\n", next_id++, id(kernel));
        std::print(md, "!hipsycl.sscp.annotations = !{{");
        for (int i = 0; i < next_id; ++i)
            std::print(md, "{}!{}", i == 0 ? "" : ", ", i);
        std::print(md, "}}\n");
    }

    // AdaptiveCpp's PTX backend flavoring appends its nvvm-reflect settings via
    // M.getModuleFlagsMetadata()->addOperand(...), which dereferences null when the module has
    // no !llvm.module.flags at all. clang-generated SSCP modules always have module flags;
    // MimIR's minimal device IR does not — emit a benign one to give the JIT a node to append to.
    std::print(md, "!{} = !{{i32 1, !\"uwtable\", i32 0}}\n", next_id);
    std::print(md, "!llvm.module.flags = !{{!{}}}\n", next_id);

    ostream() << md.str();
}

std::string DeviceEmitter::prepare() {
    auto is_kern = kernels_.contains(root());
    if (!is_kern) return Super::prepare();
    auto kernel = root();

    // No calling convention: SSCP kernels are plain functions; the JIT's backend flavoring adds
    // the target-specific kernel attributes when compiling the image.
    std::print(func_impls_, "define {} {}(", convert_ret_pi(kernel->type()->ret_pi()), id(kernel));

    auto [m1, m3, m4, m5, group_id, item_id, smem, arg, ret_lam] = kernel->vars<9>();

    auto arg_name     = id(arg);
    auto arg_type_str = convert(arg->type());
    auto arg_arity    = arg->num_projs();
    locals_[arg]      = arg_name;

    auto& bb = lam2bb_[kernel];

    // SSCP "free kernel" ABI: each logical kernel argument becomes one LLVM parameter and one
    // HCF parameter (cf. AdaptiveCpp's FreeKernelCall pass). For a tuple-typed argument the
    // aggregate is rebuilt at function entry so the body's extractvalue uses work unchanged.
    if (arg_arity > 1) {
        std::vector<std::pair<std::string, std::string>> flat_params;
        flat_params.reserve(arg_arity);
        for (size_t i = 0; i != arg_arity; ++i) {
            auto param_type = convert(arg->proj(i)->type());
            auto param_name = std::format("{}.f{}", arg_name, i);
            flat_params.emplace_back(param_type, param_name);
            std::print(func_impls_, "{}{} {}", i == 0 ? "" : ", ", param_type, param_name);
        }
        std::print(func_impls_, ") {{\n");

        std::string prev = "undef";
        for (size_t i = 0; i != arg_arity; ++i) {
            bool last      = i + 1 == arg_arity;
            auto step_name = last ? arg_name : std::format("{}.iv{}", arg_name, i);
            bb.assign(step_name, "insertvalue {} {}, {} {}, {}", arg_type_str, prev, flat_params[i].first,
                      flat_params[i].second, i);
            prev = step_name;
        }
    } else {
        std::print(func_impls_, "{} {}) {{\n", arg_type_str, arg_name);
    }

    auto register_sreg_idx = [&](const Def* def, std::string_view sscp_builtin) {
        auto name        = id(def);
        auto type        = def->type();
        auto type_name   = convert(type);
        auto opt_idx_lit = Idx::isa_lit(type);
        if (!opt_idx_lit) fe::throwf("Type of '{}' must have known index type but has {}", def, type);
        locals_[def] = name;
        declare_sscp(sscp_builtin, "i64 @{}()", sscp_builtin);
        if (type_name == "i0") {
            locals_[def] = "0";
        } else if (type_name == "i64") {
            bb.assign(name, "call i64 @{}()", sscp_builtin);
        } else {
            auto i64 = bb.assign(name + "i64", "call i64 @{}()", sscp_builtin);
            bb.assign(name, "trunc i64 {} to {}", i64, type_name);
        }
    };
    register_sreg_idx(group_id, "__acpp_sscp_get_group_id_x");
    register_sreg_idx(item_id, "__acpp_sscp_get_local_id_x");

    auto shared_as = Lit::expect(world().annex<gpu::addr_space_shared>(), "the shared address space");
    if (auto sigma = smem->type()->isa<Sigma>()) {
        if (sigma->num_ops() != 0)
            fe::throwf("ll_pcuda backend: shared-memory variable must be an empty sigma, but got '{}'", smem->type());
    } else {
        auto ptr    = Axm::expect<mem::Ptr>(smem->type(), "a shared-memory pointer type");
        auto [T, a] = ptr->args<2>();
        if (Lit::expect(a, "an address space") != shared_as)
            fe::throwf("ll_pcuda backend: shared-memory variable must live in the shared address space, but got '{}'",
                       smem->type());
        // Module-scope local-memory global; the addrspacecast yields the generic pointer the body
        // uses (cf. an acpp-built `__shared__` variable, `__acpp_local_mem.<fn>.<n>` in AS 3).
        auto name = "@" + smem->unique_name();
        std::print(vars_decls_, "{} = internal addrspace({}) global {} undef\n", name, a, convert(T));
        locals_[smem] = std::format("addrspacecast (ptr addrspace({}) {} to ptr)", shared_as, name);
    }

    return kernel->unique_name();
}

std::optional<std::string> DeviceEmitter::isa_targetspecific_intrinsic(ll::BB& bb, const Def* def) {
    auto name = id(def);

    if (auto malloc = Axm::isa<mem::malloc>(def)) {
        // SSCP device images have no usable device-side malloc/free symbols (e.g. the AMDGPU
        // JIT-link fails with `undefined symbol: malloc`). Lower per-thread heap allocations to
        // `alloca` instead — the lifetime is bounded by the kernel invocation anyway.
        if (Lit::isa(malloc->decurry()->arg(1)).value_or(0) != 0) {
            emit_unsafe(malloc->arg(0));
            auto pointee = Axm::expect<mem::Ptr>(def->proj(1)->type(), "a %mem.Ptr")->arg(0);
            return bb.assign(name, "alloca {}", convert(pointee));
        }
        return std::nullopt;
    } else if (auto free = Axm::isa<mem::free>(def)) {
        // The matching free is a no-op: alloca is released at function return.
        if (Lit::isa(free->decurry()->arg(1)).value_or(0) != 0) {
            emit_unsafe(free->arg(0));
            emit_unsafe(free->arg(1));
            return name;
        }
        return std::nullopt;
    } else if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(def)) {
        // Canonical SSCP work-group barrier; scope = work_group (2), order = relaxed (0),
        // matching pCUDA's own __syncthreads() lowering.
        declare_sscp("__acpp_sscp_work_group_barrier", "void @__acpp_sscp_work_group_barrier(i32, i32)");

        emit_unsafe(sync_work_items->arg(0));
        emit_unsafe(sync_work_items->arg(1));
        std::print(bb.body().emplace_back(), "call void @__acpp_sscp_work_group_barrier(i32 2, i32 0)");
        return name;
    }
    return std::nullopt;
}

void emit_host(World& world, std::ostream& ostream, const std::optional<HcfEmbed>& hcf, ll::Emitter::Rt rt) {
    HostEmitter emitter(world, ostream, hcf);
    emitter.rt_mode(rt);
    // Same one-liner the `ll` backend uses; each backend just names its own runtime module.
    if (rt == ll::Emitter::Rt::embed) emitter.load_rt_module("ll_pcuda_rt.ll");
    emitter.run();
}

DeviceEmitResult emit_device(World& world, std::ostream& ostream) {
    DeviceEmitter emitter(world, ostream);
    emitter.run();
    return std::move(emitter.result());
}

} // namespace mim::plug::ll_pcuda
