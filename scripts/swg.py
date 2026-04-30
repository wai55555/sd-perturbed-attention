try:
    import pag_nodes

    if pag_nodes.BACKEND in {"Forge", "reForge"}:
        import sys
        import traceback
        from functools import partial

        import gradio as gr

        from modules import scripts, script_callbacks
        from modules.ui_components import InputAccordion

        opSWG = pag_nodes.SlidingWindowGuidanceAdvanced()

        class SlidingWindowGuidanceScript(scripts.Script):
            def title(self):
                return "Sliding Window Guidance"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)
                    scale = gr.Slider(label="SWG Scale", minimum=0.0, maximum=30.0, step=0.01, value=5.0)
                    tile_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Tile Width", value=768)
                    tile_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Tile Height", value=768)
                    tile_overlap = gr.Slider(minimum=64, maximum=2048, step=8, label="Tile Overlap", value=256)

                    with InputAccordion(False, label="Override for Hires. fix") as hr_override:
                        hr_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
                        hr_scale = gr.Slider(label="SWG Scale", minimum=0.0, maximum=30.0, step=0.01, value=5.0)

                    with gr.Row():
                        sigma_start = gr.Number(minimum=-1.0, label="Sigma Start", value=-1.0)
                        sigma_end = gr.Number(minimum=-1.0, label="Sigma End", value=5.42)

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="swg_enabled" in p)),
                        (scale, "swg_scale"),
                        (tile_width, "swg_tile_width"),
                        (tile_height, "swg_tile_height"),
                        (tile_overlap, "swg_tile_overlap"),
                        (hr_override, lambda p: gr.Checkbox.update(value="swg_hr_override" in p)),
                        (hr_cfg, "swg_hr_cfg"),
                        (hr_scale, "swg_hr_scale"),
                        (sigma_start, "swg_sigma_start"),
                        (sigma_end, "swg_sigma_end"),
                    )

                return (
                    enabled,
                    scale,
                    tile_width,
                    tile_height,
                    tile_overlap,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    sigma_start,
                    sigma_end,
                )

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                (
                    enabled,
                    scale,
                    tile_width,
                    tile_height,
                    tile_overlap,
                    hr_override,
                    hr_cfg,
                    hr_scale,
                    sigma_start,
                    sigma_end,
                ) = script_args

                # Override with XYZ Plot values
                xyz = getattr(p, "_swg_xyz", {})
                if "enabled" in xyz:
                    enabled = xyz["enabled"] == "True"
                if "scale" in xyz:
                    scale = xyz["scale"]
                if "tile_width" in xyz:
                    tile_width = int(xyz["tile_width"])
                if "tile_height" in xyz:
                    tile_height = int(xyz["tile_height"])
                if "tile_overlap" in xyz:
                    tile_overlap = int(xyz["tile_overlap"])
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

                if not enabled:
                    return

                unet = p.sd_model.forge_objects.unet

                hr_enabled = getattr(p, "enable_hr", False)

                if hr_enabled and p.is_hr_pass and hr_override:
                    p.cfg_scale_before_hr = p.cfg_scale
                    p.cfg_scale = hr_cfg
                    unet = opSWG.patch(unet, hr_scale, tile_width, tile_height, tile_overlap, sigma_start, sigma_end)[0]
                else:
                    unet = opSWG.patch(unet, scale, tile_width, tile_height, tile_overlap, sigma_start, sigma_end)[0]

                p.sd_model.forge_objects.unet = unet

                p.extra_generation_params.update(
                    dict(
                        swg_enabled=enabled,
                        swg_scale=scale,
                        swg_tile_width=tile_width,
                        swg_tile_height=tile_height,
                        swg_tile_overlap=tile_overlap,
                    )
                )
                if hr_enabled:
                    p.extra_generation_params["swg_hr_override"] = hr_override
                    if hr_override:
                        p.extra_generation_params.update(
                            dict(
                                swg_hr_cfg=hr_cfg,
                                swg_hr_scale=hr_scale,
                            )
                        )
                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            swg_sigma_start=sigma_start,
                            swg_sigma_end=sigma_end,
                        )
                    )

                return

            def post_sample(self, p, ps, *script_args):
                (
                    enabled,
                    scale,
                    tile_width,
                    tile_height,
                    tile_overlap,
                    hr_override,
                    hr_cfg,
                    hr_scale,
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
            """Receive a value from XYZ Plot and store it in p._swg_xyz dict."""
            if not hasattr(p, "_swg_xyz"):
                p._swg_xyz = {}
            p._swg_xyz[field] = x

        def make_axis_on_xyz_grid():
            """Add SWG parameter axis options to XYZ Plot."""
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
                    "(SWG) Enabled",
                    str,
                    partial(set_value, field="enabled"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(SWG) Scale",
                    float,
                    partial(set_value, field="scale"),
                ),
                xyz_grid.AxisOption(
                    "(SWG) Tile Width",
                    float,
                    partial(set_value, field="tile_width"),
                ),
                xyz_grid.AxisOption(
                    "(SWG) Tile Height",
                    float,
                    partial(set_value, field="tile_height"),
                ),
                xyz_grid.AxisOption(
                    "(SWG) Tile Overlap",
                    float,
                    partial(set_value, field="tile_overlap"),
                ),
                xyz_grid.AxisOption(
                    "(SWG) Sigma Start",
                    float,
                    partial(set_value, field="sigma_start"),
                ),
                xyz_grid.AxisOption(
                    "(SWG) Sigma End",
                    float,
                    partial(set_value, field="sigma_end"),
                ),
                # Hires Fix parameters
                xyz_grid.AxisOption(
                    "(SWG) Hires Override",
                    str,
                    partial(set_value, field="hr_override"),
                    choices=lambda: ["True", "False"]
                ),
                xyz_grid.AxisOption(
                    "(SWG) Hires CFG",
                    float,
                    partial(set_value, field="hr_cfg"),
                ),
                xyz_grid.AxisOption(
                    "(SWG) Hires Scale",
                    float,
                    partial(set_value, field="hr_scale"),
                ),
            ]

            # Prevent duplicate registration
            if not any(x.label.startswith("(SWG)") for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(axis)

        def on_before_ui():
            """Register XYZ Plot axes before UI initialization."""
            try:
                make_axis_on_xyz_grid()
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] SWG Script: xyz_grid error:\n{error}",
                    file=sys.stderr,
                )

        script_callbacks.on_before_ui(on_before_ui)

except ImportError:
    pass
