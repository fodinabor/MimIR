#pragma once

#include <cstdint>

#include <optional>
#include <string>
#include <vector>

#include <mim/plug/ll/ll.h>

#include "mim/plug/ll_pcuda/phase/hcf.h"

namespace mim {

class World;

namespace plug::ll_pcuda {

namespace ll = mim::plug::ll;

/// Kernel metadata collected while emitting the device module; feeds build_hcf().
struct DeviceEmitResult {
    std::vector<HcfKernel> kernels;
    /// `__acpp_sscp_*` builtins the device module declares but does not define;
    /// AdaptiveCpp's JIT resolves them against its per-backend kernel library.
    std::vector<std::string> imported_symbols;
};

/// The serialized HCF blob to embed into the host module plus its object id.
struct HcfEmbed {
    std::string blob;
    uint64_t object_id;
};

void emit_host(World&, std::ostream&, const std::optional<HcfEmbed>&, ll::Emitter::Rt rt = ll::Emitter::Rt::embed);
DeviceEmitResult emit_device(World&, std::ostream&);

} // namespace plug::ll_pcuda

} // namespace mim
