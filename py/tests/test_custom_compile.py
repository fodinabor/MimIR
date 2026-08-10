import mim
import mim.plug.compile as compile

def test_custom_compile():
    driver = mim.Driver()
    world = driver.world()
    
    driver.load_plugins(["compile", "tensor"])
    
    phases_ax = world.annex(compile.phases.value)
    named_ax = world.annex(compile.named.value)
    phase_t = world.annex(compile.Phase.value)
    
    def to_mimir_str(s):
        ops = [world.lit_i8(ord(c)) for c in s]
        return world.tuple(ops)
        
    p1 = world.implicit_app(named_ax, to_mimir_str("%tensor.lower_tensor"))
    p2 = world.implicit_app(named_ax, to_mimir_str("%tensor.lower_map_reduce"))
    
    phases_applied = world.implicit_app(phases_ax, world.lit_ff())
    final_phase = world.implicit_app(phases_applied, world.tuple([p1, p2]))
    
    compile_lam = world.mut_lam([], phase_t)
    compile_lam.set("_compile")
    compile_lam.set_body(True, final_phase)

    compile_lam.externalize()
    
    # Run optimize!
    world.optimize()
