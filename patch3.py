import re

with open('app/viewmodels/pattern_viewmodel.py', 'r') as f:
    pv = f.read()

# 1. Update pattern_viewmodel.py __init__
pv = re.sub(
    r"(self\.linemod_matcher = LinemodMatcher\(LinemodConfig\(\)\))",
    r"self._last_params_hash = ''\n        \1",
    pv
)

# 2. Update apply_ui_configs
pv = re.sub(
    r"cfg\.T_PYRAMID\s*=\s*\[t_spread, t_spread \* 2, t_spread \* 4\]\s*cfg\.HYSTERESIS_KERNEL\s*=\s*int\(tk_vars\['pattern_hyst_var'\]\.get\(\)\)\s*cfg\.PYRAMID_LEVELS\s*=\s*3",
    r"cfg.T_PYRAMID        = [t_spread, t_spread * 2, t_spread * 4, t_spread * 8, t_spread * 16]\n            cfg.HYSTERESIS_KERNEL = int(tk_vars['pattern_hyst_var'].get())\n            cfg.PYRAMID_LEVELS   = 5",
    pv
)

# 3. Update run_find_pattern
old_run_find = '''        # Re-generate templates to ensure any updated config parameters
        # (Num Features, Weak Threshold, etc.) are applied before matching.
        self._log("[Pattern] Extracting features and building templates...")
        self.linemod_matcher.generate_templates()'''

new_run_find = '''        # Smart Verification Hash String checks if parameters changed
        current_hash = f"{tk_vars['pattern_num_var'].get()}_{tk_vars['pattern_weak_var'].get()}_{tk_vars['pattern_tspread_var'].get()}_{tk_vars['pattern_mode_var'].get()}_{tk_vars['pattern_hyst_var'].get()}"
        
        if getattr(self, '_last_params_hash', None) == current_hash:
            self._log("[Pattern] Settings matched cache. Skipping template re-generation...")
        else:
            self._log("[Pattern] Extracting features and building templates...")
            self.linemod_matcher.generate_templates()
            self._last_params_hash = current_hash'''

pv = pv.replace(old_run_find, new_run_find)

with open('app/viewmodels/pattern_viewmodel.py', 'w') as f:
    f.write(pv)


with open('app/services/linemod_matcher.py', 'r') as f:
    lm = f.read()

# 4. Update _match_pyramid
old_match = '''        # Determine the maximum pyramid level available across all templates
        max_available_level = min([len(tp['templates']) for tp in self.template_pyramids]) - 1
        top_level = min(cfg.PYRAMID_LEVELS - 1, max_available_level)'''

new_match = '''        import math
        # Dynamic Multi-Level Pyramid Engine based on max_dim / 1200px
        max_dim = max(sh, sw)
        ideal_level = int(max(1, round(math.log2(max(1, max_dim / 1200.0)))))
        
        # Determine the maximum pyramid level available across all templates
        max_available_level = min([len(tp['templates']) for tp in self.template_pyramids]) - 1
        top_level = min(ideal_level, max_available_level)
        print(f"  [Pyramid] Dynamic Engine chose Level {top_level} for {sw}x{sh} image (Max available: {max_available_level})")'''

lm = lm.replace(old_match, new_match)

with open('app/services/linemod_matcher.py', 'w') as f:
    f.write(lm)
print("Patch applied successfully.")
