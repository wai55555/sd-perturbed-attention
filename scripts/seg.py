try:
    import pag_nodes

    if pag_nodes.BACKEND in {"Forge", "reForge"}:
        import sys
        import traceback
        from functools import partial

        import gradio as gr

        from modules import scripts, script_callbacks
        from modules.ui_components import InputAccordion

        opSEG = pag_nodes.SmoothedEnergyGuidanceAdvanced()

        class SmoothedEnergyGuidanceScript(scripts.Script):
            def title(self):
                return "Smoothed Energy Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    scale = gr.Slider(label="SEG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                    with gr.Row():
                        rescale_seg = gr.Slider(label="Rescale SEG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                        rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
                    blur_sigma = gr.Slider(label="Blur Sigma", minimum=-1.0, maximum=9999.0, step=0.01, value=-1.0)
                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                        hr_scale = gr.Slider(label="SEG Scale", minimum=0.0, maximum=30.0, step=0.01, value=3.0)
                        with gr.Row():
                            hr_rescale_seg = gr.Slider(label="Rescale SEG", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            hr_rescale_mode = gr.Dropdown(choices=["full", "partial", "snf"], value="full", label="Rescale Mode")
                        hr_blur_sigma = gr.Slider(label="Blur Sigma", minimum=-1.0, maximum=9999.0, step=0.01, value=-1.0)
                    with gr.Row():
                        block = gr.Dropdown(choices=["input", "middle", "output"], value="middle", label="U-Net Block")
                        block_id = gr.Number(label="U-Net Block Id", value=0, precision=0, minimum=0)
                        block_list = gr.Text(label="U-Net Block List")
                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=-1.0)

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="seg_enabled" in p)),
                        (scale, "seg_scale"),
                        (rescale_seg, "seg_rescale"),
                        (rescale_mode, lambda p: gr.Dropdown.update(value=p.get("seg_rescale_mode", "full"))),
                        (blur_sigma, "seg_blur_sigma"),
                        (hr_override, lambda p: gr.Checkbox.update(value="seg_hr_override" in p)),
                        (hr_cfg, "seg_hr_cfg"),
                        (hr_scale, "seg_hr_scale"),
                        (hr_rescale_seg, "seg_hr_rescale"),
                        (hr_rescale_mode, lambda p: gr.Dropdown.update(value=p.get("seg_hr_rescale_mode", "full"))),
                        (hr_blur_sigma, "seg_hr_blur_sigma"),
                        (block, lambda p: gr.Dropdown.update(value=p.get("seg_block", "middle"))),
                        (block_id, "seg_block_id"),
                        (block_list, lambda p: gr.Text.update(value=p.get("seg_block_list", ""))),
                        (sigma_start, "seg_sigma_start"),
                        (sigma_end, "seg_sigma_end"),
                    )

                return enabled, scale, rescale_seg, rescale_mode, blur_sigma, block, block_id, block_list, hr_override, hr_cfg, hr_scale, hr_rescale_seg, hr_rescale_mode, hr_blur_sigma, sigma_start, sigma_end

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                (
                    enabled,
                    scale,
                    rescale_seg,
                    rescale_mode,
                    blur_sigma,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_rescale_seg,
                    hr_rescale_mode,
                    hr_blur_sigma,
                    sigma_start,
                    sigma_end,
                ) = script_args

                # Override with XYZ Plot values
                xyz = getattr(p, "_seg_xyz", {})
                if "enabled" in xyz:
                    enabled = xyz["enabled"] == "True"
                if "scale" in xyz:
                    scale = xyz["scale"]
                if "rescale_seg" in xyz:
                    rescale_seg = xyz["rescale_seg"]
                if "rescale_mode" in xyz:
                    rescale_mode = xyz["rescale_mode"]
                if "blur_sigma" in xyz:
                    blur_sigma = xyz["blur_sigma"]
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
                if "hr_rescale_seg" in xyz:
                    hr_rescale_seg = xyz["hr_rescale_seg"]
                if "hr_rescale_mode" in xyz:
                    hr_rescale_mode = xyz["hr_rescale_mode"]
                if "hr_blur_sigma" in xyz:
                    hr_blur_sigma = xyz["hr_blur_sigma"]

                if not enabled:
                    return

                unet = p.sd_model.forge_objects.unet

                hr_enabled = getattr(p, "enable_hr", False)

                if hr_enabled and p.is_hr_pass and hr_override:
                    p.cfg_scale_before_hr = p.cfg_scale
                    p.cfg_scale = hr_cfg
                    unet = opSEG.patch(unet, hr_scale, hr_blur_sigma, block, block_id, sigma_start, sigma_end, hr_rescale_seg, hr_rescale_mode, block_list)[0]
                else:
                    unet = opSEG.patch(unet, scale, blur_sigma, block, block_id, sigma_start, sigma_end, rescale_seg, rescale_mode, block_list)[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        seg_enabled=enabled,
                        seg_scale=scale,
                        seg_rescale=rescale_seg,
                        seg_rescale_mode=rescale_mode,
                        seg_blur_sigma=blur_sigma,
                        seg_block=block,
                        seg_block_id=block_id,
                        seg_block_list=block_list,
                    )
                )
                if hr_enabled:
                    p.extra_generation_params["seg_hr_override"] = hr_override
                    if hr_override:
                        p.extra_generation_params.update(
                            dict(
                                seg_hr_cfg=hr_cfg,
                                seg_hr_scale=hr_scale,
                                seg_hr_rescale=hr_rescale_seg,
                                seg_hr_rescale_mode=hr_rescale_mode,
                                seg_hr_blur_sigma=hr_blur_sigma,
                            )
                        )
                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            seg_sigma_start=sigma_start,
                            seg_sigma_end=sigma_end,
                        )
                    )

                return

            def post_sample(self, p, ps, *script_args):
                (
                    enabled,
                    scale,
                    rescale_seg,
                    rescale_mode,
                    blur_sigma,
                    block,
                    block_id,
                    block_list,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    hr_rescale_seg,
                    hr_rescale_mode,
                    hr_blur_sigma,
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
            """Receive a value from XYZ Plot and store it in p._seg_xyz dict."""
            if not hasattr(p, "_seg_xyz"):
                p._seg_xyz = {}
            p._seg_xyz[field] = x

        def make_axis_on_xyz_grid():
            """Add SEG parameter axis options to XYZ Plot."""
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
                    "(SEG) Enabled",
                    str,
                    partial(set_value, field="enabled"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(SEG) Scale",
                    float,
                    partial(set_value, field="scale"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Rescale",
                    float,
                    partial(set_value, field="rescale_seg"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Rescale Mode",
                    str,
                    partial(set_value, field="rescale_mode"),
                    choices=lambda: ["full", "partial", "snf"]
                ),
                xyz_grid.AxisOption(
                    "(SEG) Blur Sigma",
                    float,
                    partial(set_value, field="blur_sigma"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Block",
                    str,
                    partial(set_value, field="block"),
                    choices=lambda: ["input", "middle", "output"]
                ),
                xyz_grid.AxisOption(
                    "(SEG) Block Id",
                    float,
                    partial(set_value, field="block_id"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Block List",
                    str,
                    partial(set_value, field="block_list"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Sigma Start",
                    float,
                    partial(set_value, field="sigma_start"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Sigma End",
                    float,
                    partial(set_value, field="sigma_end"),
                ),
                # Hires Fix parameters
                xyz_grid.AxisOption(
                    "(SEG) Hires Override",
                    str,
                    partial(set_value, field="hr_override"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(SEG) Hires CFG",
                    float,
                    partial(set_value, field="hr_cfg"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Hires Scale",
                    float,
                    partial(set_value, field="hr_scale"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Hires Rescale",
                    float,
                    partial(set_value, field="hr_rescale_seg"),
                ),
                xyz_grid.AxisOption(
                    "(SEG) Hires Rescale Mode",
                    str,
                    partial(set_value, field="hr_rescale_mode"),
                    choices=lambda: ["full", "partial", "snf"]
                ),
                xyz_grid.AxisOption(
                    "(SEG) Hires Blur Sigma",
                    float,
                    partial(set_value, field="hr_blur_sigma"),
                ),
            ]

            # Prevent duplicate registration
            if not any(x.label.startswith("(SEG)") for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(axis)

        def on_before_ui():
            """Register XYZ Plot axes before UI initialization."""
            try:
                make_axis_on_xyz_grid()
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] SEG Script: xyz_grid error:\n{error}",
                    file=sys.stderr,
                )

        script_callbacks.on_before_ui(on_before_ui)

except ImportError:
    pass
