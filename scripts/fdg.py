try:
    import nag_forge_utils

    if nag_forge_utils.BACKEND in {"Forge", "reForge"}:
        import math
        import torch
        import torch.nn.functional as F
        import gradio as gr
        from modules import scripts
        from modules.ui_components import InputAccordion

        try:
            from guidance_utils import project
        except ImportError:
            from .guidance_utils import project

        def _build_laplacian_pyramid(x: torch.Tensor, levels: int) -> list:
            """
            Build a Laplacian pyramid.
            Equivalent implementation to kornia's build_laplacian_pyramid.
            Args:
                x: Input tensor (B, C, H, W)
                levels: Number of pyramid levels

            Returns:
                pyramid: List of Laplacian pyramid levels
            """
            try:
                from kornia.geometry import build_laplacian_pyramid
                return build_laplacian_pyramid(x, levels)
            except ImportError:
                # Fallback implementation when kornia is not available
                pyramid = []
                current = x
                for _ in range(levels):
                    # Downsample with Gaussian blur
                    blurred = F.avg_pool2d(current, kernel_size=2, stride=2, padding=0)
                    # Upsample and compute difference
                    upsampled = F.interpolate(
                        blurred, size=current.shape[-2:], mode='bilinear', align_corners=False
                    )
                    laplacian = current - upsampled
                    pyramid.append(laplacian)
                    current = blurred
                pyramid.append(current)
                return pyramid

        def _build_image_from_pyramid(pyramid: list) -> torch.Tensor:
            """
            Reconstruct an image from a Laplacian pyramid.
            Args:
                pyramid: List of Laplacian pyramid levels
            Returns:
                img: Reconstructed image tensor
            """
            img = pyramid[-1]
            for i in range(len(pyramid) - 2, -1, -1):
                try:
                    from kornia.geometry import pyrup
                    img = pyrup(img) + pyramid[i]
                except ImportError:
                    img = F.interpolate(
                        img, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False
                    ) + pyramid[i]
                del pyramid[i]
            return img

        def _get_pad_size(h: int, w: int) -> tuple:
            """
            Calculate padding size to the next power of 2.
            Args:
                h: Height
                w: Width
            Returns:
                (h_ceil, w_ceil): Padded dimensions
            """
            h_ceil = 2 ** math.ceil(math.log2(h)) if h > 1 else 1
            w_ceil = 2 ** math.ceil(math.log2(w)) if w > 1 else 1
            return h_ceil, w_ceil

        def compute_fdg_guidance(
            cond: torch.Tensor,
            uncond: torch.Tensor,
            strength_high: float = 12.0,
            strength_low: float = 1.0,
            levels: int = 2,
        ) -> torch.Tensor:
            """
            Compute FDG guidance.
            Based on paper 2506.19713 'Guidance in the Frequency Domain Enables
            High-Fidelity Sampling at Low CFG Scales'.
            Args:
                cond: Conditional prediction tensor
                uncond: Unconditional prediction tensor
                strength_high: High frequency guidance strength
                strength_low: Low frequency guidance strength (equivalent to CFG scale)
                levels: Number of Laplacian pyramid levels

            Returns:
                guidance_diff: FDG guided prediction
            """
            height, width = cond.shape[2:4]
            h_ceil, w_ceil = _get_pad_size(height, width)

            # Pad if necessary
            needs_pad = (h_ceil != height or w_ceil != width)
            if needs_pad:
                pad_h = h_ceil - height
                pad_w = w_ceil - width
                cond = F.pad(cond, (0, pad_w, 0, pad_h))
                uncond = F.pad(uncond, (0, pad_w, 0, pad_h))

            # Build Laplacian pyramids
            cond_pyramid = _build_laplacian_pyramid(cond, levels)
            uncond_pyramid = _build_laplacian_pyramid(uncond, levels)

            guided_pyramid: list = []
            scales = [strength_high, strength_low]

            for i, (cond_i, uncond_i) in enumerate(zip(cond_pyramid, uncond_pyramid)):
                # Compute guidance diff in frequency band
                diff = cond_i - uncond_i

                # Decompose into parallel and orthogonal components (matching ComfyUI implementation)
                diff_parallel, diff_orthogonal = project(diff, cond_i)
                diff = diff_parallel + diff_orthogonal

                # Select scale (high frequency or low frequency)
                scale = scales[min(i, len(scales) - 1)]

                # Compute guided image
                guided_i = cond_i + (scale - 1.0) * diff
                guided_pyramid.append(guided_i)

            guidance_diff = _build_image_from_pyramid(guided_pyramid)

            # Remove padding
            if needs_pad:
                guidance_diff = guidance_diff[:, :, :height, :width]

            return guidance_diff

        class FrequencyDecoupledGuidanceScript(scripts.Script):
            """FDG (Frequency-Decoupled Guidance) script for reForge/Forge"""

            def title(self) -> str:
                return "Frequency-Decoupled Guidance"

            def show(self, is_img2img: bool):
                return scripts.AlwaysVisible

            def ui(self, *args, **kwargs):
                with gr.Accordion(open=False, label=self.title()):
                    enabled = gr.Checkbox(label="Enabled", value=False)

                    strength_high = gr.Slider(
                        label="Strength High (High Freq)",
                        minimum=0.0,
                        maximum=50.0,
                        step=0.1,
                        value=12.0,
                        info="High frequency guidance strength. Paper recommended: 12.0"
                    )

                    strength_low = gr.Slider(
                        label="Strength Low (Low Freq / CFG)",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=1.0,
                        info="Low frequency guidance strength (equivalent to CFG scale). Recommended 1.0 for low-CFG workflow"
                    )

                    with InputAccordion(False, label="Override for Hires Fix") as hr_override:
                        hr_strength_high = gr.Slider(
                            label="Strength High (High Freq)",
                            minimum=0.0,
                            maximum=50.0,
                            step=0.1,
                            value=12.0
                        )
                        hr_strength_low = gr.Slider(
                            label="Strength Low (Low Freq / CFG)",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=1.0
                        )

                    self.infotext_fields = (
                        (enabled, lambda p: gr.Checkbox.update(value="fdg_enabled" in p)),
                        (strength_high, "fdg_strength_high"),
                        (strength_low, "fdg_strength_low"),
                        (hr_override, lambda p: gr.Checkbox.update(value="fdg_hr_override" in p)),
                        (hr_strength_high, "fdg_hr_strength_high"),
                        (hr_strength_low, "fdg_hr_strength_low"),
                    )

                return (
                    enabled,
                    strength_high,
                    strength_low,
                    hr_override,
                    hr_strength_high,
                    hr_strength_low,
                )

            def process_before_every_sampling(self, p, *script_args, **kwargs):
                """Patch UNet with FDG guidance before sampling."""
                (
                    enabled,
                    strength_high,
                    strength_low,
                    hr_override,
                    hr_strength_high,
                    hr_strength_low,
                ) = script_args

                if not enabled:
                    return

                hr_enabled = getattr(p, "enable_hr", False)
                is_hr_pass = getattr(p, "is_hr_pass", False)

                # Hires Fix override
                if hr_enabled and is_hr_pass and hr_override:
                    active_strength_high = hr_strength_high
                    active_strength_low = hr_strength_low
                else:
                    active_strength_high = strength_high
                    active_strength_low = strength_low

                unet = p.sd_model.forge_objects.unet
                unet = unet.clone()

                # Patch CFG function
                _strength_high = active_strength_high
                _strength_low = active_strength_low

                def fdg_cfg_function(args):
                    """FDG guidance CFG function"""
                    cond_denoised: torch.Tensor = args["cond_denoised"]
                    uncond_denoised: torch.Tensor = args["uncond_denoised"]
                    x_orig: torch.Tensor = args["input"]

                    # Compute FDG guidance result
                    fdg_result = compute_fdg_guidance(
                        cond_denoised,
                        uncond_denoised,
                        strength_high=_strength_high,
                        strength_low=_strength_low,
                    )

                    # Return guidance diff (matching ComfyUI implementation)
                    return x_orig - fdg_result

                unet.model_options["sampler_cfg_function"] = fdg_cfg_function

                p.sd_model.forge_objects.unet = unet

                # Record metadata
                p.extra_generation_params.update(
                    dict(
                        fdg_enabled=enabled,
                        fdg_strength_high=strength_high,
                        fdg_strength_low=strength_low,
                    )
                )

                if hr_enabled and hr_override:
                    p.extra_generation_params.update(
                        dict(
                            fdg_hr_override=hr_override,
                            fdg_hr_strength_high=hr_strength_high,
                            fdg_hr_strength_low=hr_strength_low,
                        )
                    )

except ImportError:
    pass
