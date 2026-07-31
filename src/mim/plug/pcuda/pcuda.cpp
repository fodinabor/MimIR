#include "mim/plug/pcuda/pcuda.h"

#include <mim/phase.h>
#include <mim/plugin.h>

#include "mim/plug/gpu/gpu.h"

using namespace mim;
using namespace mim::plug;

void reg_phases(Flags2Phases& phases) {
    MIM_REPL(phases, pcuda::stream_impl_repl, {
        auto stream_flags = Annex::base<gpu::Stream>();
        if (def->flags() == stream_flags) return world().annex<pcuda::Stream>();
        return {};
    });
}

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"pcuda", MIM_VERSION, nullptr, reg_phases}; }
