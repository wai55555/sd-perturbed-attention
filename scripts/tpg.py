try:
    import nag_forge_utils

    if nag_forge_utils.BACKEND in {"Forge", "reForge"}:
        import torch
        import gradio as gr
        from modules import scripts
        from modules.ui_components import InputAccordion

        if nag_forge_utils.BACKEND == "reForge":
            from ldm_patched.ldm.modules.attention import BasicTransformerBlock
            from ldm_patched.modules.samplers import calc_cond_uncond_batch
        else:
            from backend.nn.unet import BasicTransformerBlock
            from backend.sampling.sampling_function import calc_cond_uncond_batch

        try:
            from guidance_utils import rescale_guidance, set_model_options_value, snf_guidance
        except ImportError:
            from .guidance_utils import rescale_guidance, set_model_options_value, snf_guidance

        TPG_OPTION = "tpg"

        def shuffle_tokens(x: torch.Tensor) -> torch.Tensor:
            """Randomly shuffle tokens."""
            permutation = torch.randperm(x.shape[1], device=x.device)
            return x[:, permutation]

        def tpg_forward_wrapper(forward_orig):
            """Wrap BasicTransformerBlock.forward with TPG flag support."""
            @torch.no_grad()
            def forward(x: torch.Tensor, context=None, transformer_options: dict = {}):
                is_tpg = transformer_options.get(TPG_OPTION, False)
                x_tpg = shuffle_tokens(x) if is_tpg else x
                return forward_orig(x_tpg, context=context, transformer_options=transformer_options)
            return forward


        class TokenPerturbationGuidanceScript(scripts.Script):
            """TPG (Token Perturbation Guidance) script for reForge/Forge"""

            def title(self) -> str:
                return "Token Perturbation Guidance"

            def show(self, is_img2img: bool):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)

                    scale = gr.Slider(
                        label="Scale",
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        value=3.0,
                        info="Guidance scale. Recommended value: 3.0"
                    )

                    with gr.Row():
                        sigma_start = gr.Slider(
                            label="Sigma Start",
                            minimum=-1.0,
                            maximum=10000.0,
                            step=0.01,
                            value=-1.0,
                            info="-1.0 = No limit"
                        )
                        sigma_end = gr.Slider(
                            label="Sigma End",
                            minimum=-1.0,
                            maximum=10000.0,
                            step=0.01,
                            value=-1.0,
                            info="-1.0 = No limit"
                        )

                    with gr.Row():
                        rescale = gr.Slider(
                            label="Rescale",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.0,
                            info="Rescale strength. 0.0 = disabled"
                        )
                        rescale_mode = gr.Dropdown(
                            label="Rescale Mode",
                            choices=["full", "partial", "snf"],
                            value="full"
                        )

                    unet_block_list = gr.Textbox(
                        label="U-Net Block List",
                        value="",
                        placeholder="e.g. d2.2-9,d3 (empty = all blocks)",
                        info="Target U-Net blocks for TPG. Leave empty to apply to all blocks."
                    )

                    with InputAccordion(False, label="Override for Hires Fix") as hr_override:
                        hr_scale = gr.Slider(
                            label="Scale",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=3.0
                        )

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="tpg_enabled" in p)),
                        (scale, "tpg_scale"),
                        (rescale, "tpg_rescale"),
                        (rescale_mode, "tpg_rescale_mode"),
                        (sigma_start, "tpg_sigma_start"),
                        (sigma_end, "tpg_sigma_end"),
                        (hr_override, lambda p: gr.Checkbox.update(value="tpg_hr_override" in p)),
                        (hr_scale, "tpg_hr_scale"),
                    )

                return (
                    enabled,
                    scale,
                    sigma_start,
                    sigma_end,
                    rescale,
                    rescale_mode,
                    unet_block_list,
                    hr_override,
                    hr_scale,
                )


            def process_before_every_sampling(self, p, *script_args, **kwargs):
                """Patch UNet with TPG guidance before sampling."""
                (
                    enabled,
                    scale,
                    sigma_start,
                    sigma_end,
                    rescale,
                    rescale_mode,
                    unet_block_list,
                    hr_override,
                    hr_scale,
                ) = script_args

                if not enabled:
                    return

                hr_enabled = getattr(p, "enable_hr", False)
                is_hr_pass = getattr(p, "is_hr_pass", False)

                # Hires Fix override
                if hr_enabled and is_hr_pass and hr_override:
                    active_scale = hr_scale
                else:
                    active_scale = scale

                # sigma_start of -1.0 means no limit
                _sigma_start = float("inf") if sigma_start < 0 else sigma_start
                _sigma_end = sigma_end

                unet = p.sd_model.forge_objects.unet
                unet = unet.clone()

                # Patch BasicTransformerBlock.forward with TPG wrapper
                inner_model = unet.model
                if unet_block_list:
                    try:
                        from guidance_utils import parse_unet_blocks
                    except ImportError:
                        from .guidance_utils import parse_unet_blocks
                    _, block_names = parse_unet_blocks(unet, unet_block_list, None)
                else:
                    block_names = None

                for name, module in inner_model.diffusion_model.named_modules():
                    if isinstance(module, BasicTransformerBlock):
                        if block_names is None or name in block_names:
                            forward_orig = module.forward
                            forward_tpg = tpg_forward_wrapper(forward_orig)
                            unet.add_object_patch(
                                f"diffusion_model.{name}.forward", forward_tpg
                            )

                # Set up and register post_cfg_function
                _active_scale = active_scale
                _rescale = rescale
                _rescale_mode = rescale_mode

                def tpg_post_cfg_function(args):
                    """CFG + TPG guidance calculation."""
                    model = args["model"]
                    cond_pred = args["cond_denoised"]
                    uncond_pred = args["uncond_denoised"]
                    cond = args["cond"]
                    cfg_result = args["denoised"]
                    sigma = args["sigma"]
                    model_options = args["model_options"].copy()
                    x = args["input"]

                    if _active_scale == 0 or not (_sigma_end < sigma[0] <= _sigma_start):
                        return cfg_result

                    # Enable TPG flag and recompute cond prediction
                    set_model_options_value(model_options, TPG_OPTION, True)
                    (tpg_cond_pred, _) = calc_cond_uncond_batch(
                        model, cond, None, x, sigma, model_options
                    )

                    tpg = (cond_pred - tpg_cond_pred) * _active_scale

                    if _rescale_mode == "snf":
                        if uncond_pred.any():
                            return uncond_pred + snf_guidance(cfg_result - uncond_pred, tpg)
                        return cfg_result + tpg

                    return cfg_result + rescale_guidance(
                        tpg, cond_pred, cfg_result, _rescale, _rescale_mode
                    )

                unet.set_model_sampler_post_cfg_function(
                    tpg_post_cfg_function, rescale_mode == "snf"
                )

                p.sd_model.forge_objects.unet = unet

                # Record metadata
                p.extra_generation_params.update(
                    dict(
                        tpg_enabled=enabled,
                        tpg_scale=scale,
                        tpg_rescale=rescale,
                        tpg_rescale_mode=rescale_mode,
                    )
                )

                if sigma_start >= 0 or sigma_end >= 0:
                    p.extra_generation_params.update(
                        dict(
                            tpg_sigma_start=sigma_start,
                            tpg_sigma_end=sigma_end,
                        )
                    )

                if hr_enabled and hr_override:
                    p.extra_generation_params.update(
                        dict(
                            tpg_hr_override=hr_override,
                            tpg_hr_scale=hr_scale,
                        )
                    )

except ImportError:
    pass
