add_mim_plugin(ll_pcuda
    SOURCES
        ll_pcuda.cpp
        phase/hcf.cpp
        phase/ll_pcuda.cpp
    INSTALL
)

add_mim_runtime(ll_pcuda
    SOURCES
        rt/mim_pcuda_rt.c
    INSTALL
)

# The HCF infrastructure is reused directly from AdaptiveCpp's (header-only) hcf_container.hpp.
# HIPSYCL_TOOL_COMPONENT keeps that header standalone — no link against libacpp-rt is needed
# (same setup as AdaptiveCpp's own acpp-hcf-tool); generated host modules link acpp-rt instead.
get_target_property(MIM_ACPP_INCLUDE_DIRS AdaptiveCpp::acpp-common INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(mim_ll_pcuda
    SYSTEM PRIVATE
        ${MIM_ACPP_INCLUDE_DIRS}
)
target_compile_definitions(mim_ll_pcuda
    PRIVATE
        HIPSYCL_TOOL_COMPONENT
)
