try:
    import nag_forge_utils

    if nag_forge_utils.BACKEND in {"Forge", "reForge"}:
        import gradio as gr
        from modules import scripts, script_callbacks
        from modules.ui_components import InputAccordion
        from functools import partial
        import sys
        import traceback

        opNormalizedAttention = nag_forge_utils.NormalizedAttentionGuidance()

        class NAGPreset:
            def __init__(self, scale=2.0, tau=2.5, alpha=0.5, sigma_start=-1.0, sigma_end=-1.0):
                self.scale = scale
                self.tau = tau
                self.alpha = alpha
                self.sigma_start = sigma_start
                self.sigma_end = sigma_end

        # Defined presets
        NAG_PRESETS = {
            "SDXL": NAGPreset(scale=4, tau=2.5, alpha=0.8, sigma_start=-1, sigma_end=-1.0),
            "SD1.5": NAGPreset(scale=5.0, tau=2.5, alpha=0.4, sigma_start=-1.0, sigma_end=-1.0),
            "Vanilla": NAGPreset(scale=2.0, tau=2.5, alpha=0.5, sigma_start=-1.0, sigma_end=-1.0),
        }

        class NormalizedAttentionGuidanceScript(scripts.Script):
            def title(self):
                return "Normalized Attention Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)

                    # Preset buttons
                    with gr.Group():
                        gr.Markdown("**Presets:**")
                        with gr.Row():
                            preset_buttons = {}
                            for preset_name in NAG_PRESETS:
                                preset_buttons[preset_name] = gr.Button(preset_name, size="sm", scale=1)

                    negative = gr.Textbox(
                        label="NAG Negative Prompt",
                        placeholder="Supplements CFG when scale>1, or works alone when CFG=1. Leave empty to reuse main negative.",
                        lines=2
                    )

                    scale = gr.Slider(
                        label="NAG Scale",
                        minimum=0.0,
                        maximum=50.0,
                        step=0.1,
                        value=6.0,
                        info="Typical range 1-12. Can supplement CFG or work alone"
                    )

                    tau = gr.Slider(
                        label="Tau (Normalization Threshold)",
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=2.5,
                        info="Normalization threshold, larger value should increase scale impact"
                    )

                    alpha = gr.Slider(
                        label="Alpha",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                        value=0.5,
                        info="Linear interpolation between original (at alpha=0) and NAG (at alpha=1) results"
                    )

                    hr_mode = gr.Radio(
                        show_label=False,
                        label="Hires Fix Mode",
                        choices=["Both", "HRFix Off", "HRFix Only"],
                        value="Both",
                        info="Control when NAG is active during generation",
                    )

                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_scale = gr.Slider(
                            label="NAG Scale",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=2.0
                        )
                        hr_tau = gr.Slider(
                            label="Tau (Normalization Threshold)",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=2.5
                        )
                        hr_alpha = gr.Slider(
                            label="Alpha",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.001,
                            value=0.5
                        )

                    with gr.Accordion("Advanced Settings", open=False):
                        gr.Markdown("**Sigma Range (Denoising Strength Bounds)**")
                        with gr.Row():
                            sigma_start = gr.Number(
                                minimum=-1.0,
                                label="Sigma Start",
                                value=-1.0,
                                info="Upper sigma bound where NAG is active (-1 = no limit)"
                            )
                            sigma_end = gr.Number(
                                minimum=-1.0,
                                label="Sigma End",
                                value=-1.0,
                                info="Lower sigma bound where NAG is active (-1 = no limit)"
                            )

                        gr.Markdown("**Block Selection**")
                        unet_block_list = gr.Text(
                            label="U-Net Block Selection",
                            placeholder="e.g., d5, m0, u0 | SDXL all: d0-3, m0, u0-5 (leave empty for all blocks)",
                            info="Target specific U-Net blocks. Format: d=down, m=middle, u=up. Example: 'd2-3, m0' targets later downsampling + middle.",
                            value=""
                        )

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="nag_enabled" in p)),
                        (negative, "nag_negative"),
                        (scale, "nag_scale"),
                        (tau, "nag_tau"),
                        (alpha, "nag_alpha"),
                        (hr_mode, "nag_hr_mode"),
                        (hr_override, lambda p: gr.Checkbox.update(value="nag_hr_override" in p)),
                        (hr_scale, "nag_hr_scale"),
                        (hr_tau, "nag_hr_tau"),
                        (hr_alpha, "nag_hr_alpha"),
                        (sigma_start, "nag_sigma_start"),
                        (sigma_end, "nag_sigma_end"),
                        (unet_block_list, lambda p: gr.Text.update(value=p.get("nag_block_list", ""))),
                    )

                    # Preset callback to load a preset on a button click
                    def create_preset_handler(preset_name):
                        def handler():
                            preset = NAG_PRESETS[preset_name]
                            return [
                                gr.update(value=preset.scale),
                                gr.update(value=preset.tau),
                                gr.update(value=preset.alpha),
                                gr.update(value=preset.sigma_start),
                                gr.update(value=preset.sigma_end),
                            ]
                        return handler

                    for preset_name, button in preset_buttons.items():
                        button.click(
                            fn=create_preset_handler(preset_name),
                            inputs=[],
                            outputs=[scale, tau, alpha, sigma_start, sigma_end]
                        )

                return enabled, negative, scale, tau, alpha, hr_mode, hr_override, hr_scale, hr_tau, hr_alpha, sigma_start, sigma_end, unet_block_list

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                (
                    enabled,
                    negative,
                    scale,
                    tau,
                    alpha,
                    hr_mode,
                    hr_override,
                    hr_scale,
                    hr_tau,
                    hr_alpha,
                    sigma_start,
                    sigma_end,
                    unet_block_list,
                ) = script_args

                # Override with XYZ Plot values
                xyz = getattr(p, "_nag_xyz", {})
                if "enabled" in xyz:
                    enabled = xyz["enabled"] == "True"
                if "negative" in xyz:
                    negative = xyz["negative"]
                if "scale" in xyz:
                    scale = xyz["scale"]
                if "tau" in xyz:
                    tau = xyz["tau"]
                if "alpha" in xyz:
                    alpha = xyz["alpha"]
                if "hr_mode" in xyz:
                    hr_mode = xyz["hr_mode"]
                if "hr_override" in xyz:
                    hr_override = xyz["hr_override"] == "True"
                if "hr_scale" in xyz:
                    hr_scale = xyz["hr_scale"]
                if "hr_tau" in xyz:
                    hr_tau = xyz["hr_tau"]
                if "hr_alpha" in xyz:
                    hr_alpha = xyz["hr_alpha"]
                if "sigma_start" in xyz:
                    sigma_start = xyz["sigma_start"]
                if "sigma_end" in xyz:
                    sigma_end = xyz["sigma_end"]
                if "unet_block_list" in xyz:
                    unet_block_list = xyz["unet_block_list"]

                if not enabled:
                    return

                if not negative or negative.strip() == "":
                    # Get the main negative prompt if possible
                    negative = getattr(p, 'negative_prompt', None)

                    if not negative or (isinstance(negative, list) and not negative[0]):
                        return

                    if isinstance(negative, list):
                        negative = negative[0]

                negative_cond = p.sd_model.get_learned_conditioning([negative] * p.batch_size)

                unet = p.sd_model.forge_objects.unet
                hr_enabled = getattr(p, "enable_hr", False)
                is_hr_pass = getattr(p, "is_hr_pass", False)

                # hr_mode to allow for targeting first, second, or both passes.
                if hr_mode == "HRFix Off" and is_hr_pass:
                    return  # Skip NAG on hires pass
                elif hr_mode == "HRFix Only" and not is_hr_pass:
                    return  # Skip NAG on first pass

                if hr_enabled and is_hr_pass and hr_override:
                    unet = opNormalizedAttention.patch(
                        unet,
                        negative_cond,
                        hr_scale,
                        hr_tau,
                        hr_alpha,
                        sigma_start,
                        sigma_end,
                        unet_block_list
                    )
                else:
                    unet = opNormalizedAttention.patch(
                        unet,
                        negative_cond,
                        scale,
                        tau,
                        alpha,
                        sigma_start,
                        sigma_end,
                        unet_block_list
                    )

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        nag_enabled=enabled,
                        nag_negative=negative,
                        nag_scale=scale,
                        nag_tau=tau,
                        nag_alpha=alpha,
                    )
                )

                if hr_mode != "Both":
                    p.extra_generation_params["nag_hr_mode"] = hr_mode

                if unet_block_list:
                    p.extra_generation_params["nag_block_list"] = unet_block_list

                if hr_enabled:
                    p.extra_generation_params["nag_hr_override"] = hr_override
                    if hr_override:
                        p.extra_generation_params.update(
                            dict(
                                nag_hr_scale=hr_scale,
                                nag_hr_tau=hr_tau,
                                nag_hr_alpha=hr_alpha,
                            )
                        )

                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            nag_sigma_start=sigma_start,
                            nag_sigma_end=sigma_end,
                        )
                    )

                return

        # XYZ Plot support
        def set_value(p, x, xs, *, field: str):
            """Receive a value from XYZ Plot and store it in p._nag_xyz dict."""
            if not hasattr(p, "_nag_xyz"):
                p._nag_xyz = {}
            p._nag_xyz[field] = x

        def make_axis_on_xyz_grid():
            """Add NAG parameter axis options to XYZ Plot."""
            xyz_grid = None
            for script in scripts.scripts_data:
                if script.script_class.__module__ == "xyz_grid.py":
                    xyz_grid = script.module
                    break

            if xyz_grid is None:
                return

            axis = [
                # Basic parameters
                xyz_grid.AxisOption(
                    "(NAG) Enabled",
                    str,
                    partial(set_value, field="enabled"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(NAG) Scale",
                    float,
                    partial(set_value, field="scale"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) Tau",
                    float,
                    partial(set_value, field="tau"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) Alpha",
                    float,
                    partial(set_value, field="alpha"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) Negative Prompt",
                    str,
                    partial(set_value, field="negative"),
                ),

                # Hires Fix parameters
                xyz_grid.AxisOption(
                    "(NAG) Hires Fix Mode",
                    str,
                    partial(set_value, field="hr_mode"),
                    choices=lambda: ["Both", "HRFix Off", "HRFix Only"]
                ),
                xyz_grid.AxisOption(
                    "(NAG) Hires Override",
                    str,
                    partial(set_value, field="hr_override"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(NAG) Hires Scale",
                    float,
                    partial(set_value, field="hr_scale"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) Hires Tau",
                    float,
                    partial(set_value, field="hr_tau"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) Hires Alpha",
                    float,
                    partial(set_value, field="hr_alpha"),
                ),

                # Advanced parameters
                xyz_grid.AxisOption(
                    "(NAG) Sigma Start",
                    float,
                    partial(set_value, field="sigma_start"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) Sigma End",
                    float,
                    partial(set_value, field="sigma_end"),
                ),
                xyz_grid.AxisOption(
                    "(NAG) U-Net Blocks",
                    str,
                    partial(set_value, field="unet_block_list"),
                ),
            ]

            # Prevent duplicate registration
            if not any(x.label.startswith("(NAG)") for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(axis)

        def on_before_ui():
            """Register XYZ Plot axes before UI initialization."""
            try:
                make_axis_on_xyz_grid()
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] NAG Script: xyz_grid error:\n{error}",
                    file=sys.stderr,
                )

        script_callbacks.on_before_ui(on_before_ui)

except ImportError:
    pass
