try:
    import pag_nodes

    if pag_nodes.BACKEND in {"Forge", "reForge"}:
        import sys
        import traceback
        from functools import partial

        import gradio as gr

        from modules import scripts, script_callbacks
        from modules.ui_components import InputAccordion

        opPerturbedAttention = pag_nodes.PerturbedAttention()

        class PerturbedAttentionScript(scripts.Script):
            def title(self):
                return "Perturbed-Attention Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                    with gr.Row():
                        rescale_pag = gr.Slider(label="Rescale PAG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
                    adaptive_scale = gr.Slider(label="Adaptive Scale", minimum=0.0, maximum=1.0, step=0.001, value=0.0)

                    hr_mode = gr.Radio(
                        show_label=False,
                        label="Hires Fix Mode",
                        choices=["Both", "HRFix Off", "HRFix Only"],
                        value="Both",
                        info="Control when PAG is active during generation",
                    )

                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                        hr_scale = gr.Slider(label="PAG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                        with gr.Row():
                            hr_rescale_pag = gr.Slider(label="Rescale PAG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            hr_rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
                        hr_adaptive_scale = gr.Slider(label="Adaptive Scale", minimum=0.0, maximum=1.0, step=0.001, value=0.0)
                    with gr.Row():
                        block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                        block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)
                        block_list = gr.Text(label="U-Net Block List")
                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="pag_enabled" in p)),
                        (scale, "pag_scale"),
                        (rescale_pag, "pag_rescale"),
                        (rescale_mode, lambda p: gr.Dropdown.update(value=p.get("pag_rescale_mode", "full"))),
                        (adaptive_scale, "pag_adaptive_scale"),
                        (hr_mode, "pag_hr_mode"),
                        (hr_override, lambda p: gr.Checkbox.update(value="pag_hr_override" in p)),
                        (hr_cfg, "pag_hr_cfg"),
                        (hr_scale, "pag_hr_scale"),
                        (hr_rescale_pag, "pag_hr_rescale"),
                        (hr_rescale_mode, lambda p: gr.Dropdown.update(value=p.get("pag_hr_rescale_mode", "full"))),
                        (hr_adaptive_scale, "pag_hr_adaptive_scale"),
                        (block, lambda p: gr.Dropdown.update(value=p.get("pag_block", "middle"))),
                        (block_id, "pag_block_id"),
                        (block_list, lambda p: gr.Text.update(value=p.get("pag_block_list", ""))),
                        (sigma_start, "pag_sigma_start"),
                        (sigma_end, "pag_sigma_end"),
                    )

                return enabled, scale, rescale_pag, rescale_mode, adaptive_scale, hr_mode, block, block_id, block_list, hr_override, hr_cfg, hr_scale, hr_rescale_pag, hr_rescale_mode, hr_adaptive_scale, sigma_start, sigma_end

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                (
                    enabled,
                    scale,
                    rescale_pag,
                    rescale_mode,
                    adaptive_scale,
                    hr_mode,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_rescale_pag,
                    hr_rescale_mode,
                    hr_adaptive_scale,
                    sigma_start,
                    sigma_end,
                ) = script_args

                # Override with XYZ Plot values
                xyz = getattr(p, "_pag_xyz", {})
                if "enabled" in xyz:
                    enabled = xyz["enabled"] == "True"
                if "scale" in xyz:
                    scale = xyz["scale"]
                if "rescale_pag" in xyz:
                    rescale_pag = xyz["rescale_pag"]
                if "rescale_mode" in xyz:
                    rescale_mode = xyz["rescale_mode"]
                if "adaptive_scale" in xyz:
                    adaptive_scale = xyz["adaptive_scale"]
                if "hr_mode" in xyz:
                    hr_mode = xyz["hr_mode"]
                if "block" in xyz:
                    block = xyz["block"]
                if "block_id" in xyz:
                    block_id = xyz["block_id"]
                if "block_list" in xyz:
                    block_list = xyz["block_list"]
                if "sigma_start" in xyz:
                    sigma_start = xyz["sigma_start"]
                if "sigma_end" in xyz:
                    sigma_end = xyz["sigma_end"]
                if "hr_override" in xyz:
                    hr_override = xyz["hr_override"] == "True"
                if "hr_cfg" in xyz:
                    hr_cfg = xyz["hr_cfg"]
                if "hr_scale" in xyz:
                    hr_scale = xyz["hr_scale"]
                if "hr_rescale_pag" in xyz:
                    hr_rescale_pag = xyz["hr_rescale_pag"]
                if "hr_rescale_mode" in xyz:
                    hr_rescale_mode = xyz["hr_rescale_mode"]
                if "hr_adaptive_scale" in xyz:
                    hr_adaptive_scale = xyz["hr_adaptive_scale"]

                if not enabled:
                    return

                unet = p.sd_model.forge_objects.unet
                hr_enabled = getattr(p, "enable_hr", False)
                is_hr_pass = getattr(p, "is_hr_pass", False)

                # hr_mode to allow for targeting first, second, or both passes.
                if hr_mode == "HRFix Off" and is_hr_pass:
                    return  # Skip PAG on hires pass
                elif hr_mode == "HRFix Only" and not is_hr_pass:
                    return  # Skip PAG on first pass

                if hr_enabled and is_hr_pass and hr_override:
                    p.cfg_scale_before_hr = p.cfg_scale
                    p.cfg_scale = hr_cfg
                    unet = opPerturbedAttention.patch(unet, hr_scale, hr_adaptive_scale, block, block_id, sigma_start, sigma_end, hr_rescale_pag, hr_rescale_mode, block_list)[0]
                else:
                    unet = opPerturbedAttention.patch(unet, scale, adaptive_scale, block, block_id, sigma_start, sigma_end, rescale_pag, rescale_mode, block_list)[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        pag_enabled=enabled,
                        pag_scale=scale,
                        pag_rescale=rescale_pag,
                        pag_rescale_mode=rescale_mode,
                        pag_adaptive_scale=adaptive_scale,
                        pag_block=block,
                        pag_block_id=block_id,
                        pag_block_list=block_list,
                    )
                )

                if hr_mode != "Both":
                    p.extra_generation_params["pag_hr_mode"] = hr_mode
                if hr_enabled:
                    p.extra_generation_params["pag_hr_override"] = hr_override
                    if hr_override:
                        p.extra_generation_params.update(
                            dict(
                                pag_hr_cfg=hr_cfg,
                                pag_hr_scale=hr_scale,
                                pag_hr_rescale=hr_rescale_pag,
                                pag_hr_rescale_mode=hr_rescale_mode,
                                pag_hr_adaptive_scale=hr_adaptive_scale,
                            )
                        )
                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            pag_sigma_start=sigma_start,
                            pag_sigma_end=sigma_end,
                        )
                    )

                return

            def post_sample(self, p, ps, *script_args):
                (
                    enabled,
                    scale,
                    rescale_pag,
                    rescale_mode,
                    adaptive_scale,
                    hr_mode,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_rescale_pag,
                    hr_rescale_mode,
                    hr_adaptive_scale,
                    sigma_start,
                    sigma_end,
                ) = script_args

                if not enabled:
                    return

                hr_enabled = getattr(p, "enable_hr", False)

                if hr_enabled and hr_override:
                    p.cfg_scale = p.cfg_scale_before_hr

                return

        # XYZ Plot support
        def set_value(p, x, xs, *, field: str):
            """Receive a value from XYZ Plot and store it in p._pag_xyz dict."""
            if not hasattr(p, "_pag_xyz"):
                p._pag_xyz = {}
            p._pag_xyz[field] = x

        def make_axis_on_xyz_grid():
            """Add PAG parameter axis options to XYZ Plot."""
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
                    "(PAG) Enabled",
                    str,
                    partial(set_value, field="enabled"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(PAG) Scale",
                    float,
                    partial(set_value, field="scale"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Rescale",
                    float,
                    partial(set_value, field="rescale_pag"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Rescale Mode",
                    str,
                    partial(set_value, field="rescale_mode"),
                    choices=lambda: ["full", "partial", "snf"]
                ),
                xyz_grid.AxisOption(
                    "(PAG) Adaptive Scale",
                    float,
                    partial(set_value, field="adaptive_scale"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Hires Fix Mode",
                    str,
                    partial(set_value, field="hr_mode"),
                    choices=lambda: ["Both", "HRFix Off", "HRFix Only"]
                ),
                xyz_grid.AxisOption(
                    "(PAG) Block",
                    str,
                    partial(set_value, field="block"),
                    choices=lambda: ["input", "middle", "output"]
                ),
                xyz_grid.AxisOption(
                    "(PAG) Block Id",
                    float,
                    partial(set_value, field="block_id"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Block List",
                    str,
                    partial(set_value, field="block_list"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Sigma Start",
                    float,
                    partial(set_value, field="sigma_start"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Sigma End",
                    float,
                    partial(set_value, field="sigma_end"),
                ),
                # Hires Fix parameters
                xyz_grid.AxisOption(
                    "(PAG) Hires Override",
                    str,
                    partial(set_value, field="hr_override"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(PAG) Hires CFG",
                    float,
                    partial(set_value, field="hr_cfg"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Hires Scale",
                    float,
                    partial(set_value, field="hr_scale"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Hires Rescale",
                    float,
                    partial(set_value, field="hr_rescale_pag"),
                ),
                xyz_grid.AxisOption(
                    "(PAG) Hires Rescale Mode",
                    str,
                    partial(set_value, field="hr_rescale_mode"),
                    choices=lambda: ["full", "partial", "snf"]
                ),
                xyz_grid.AxisOption(
                    "(PAG) Hires Adaptive Scale",
                    float,
                    partial(set_value, field="hr_adaptive_scale"),
                ),
            ]

            # Prevent duplicate registration
            if not any(x.label.startswith("(PAG)") for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(axis)

        def on_before_ui():
            """Register XYZ Plot axes before UI initialization."""
            try:
                make_axis_on_xyz_grid()
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] PAG Script: xyz_grid error:\n{error}",
                    file=sys.stderr,
                )

        script_callbacks.on_before_ui(on_before_ui)

except ImportError:
    pass
