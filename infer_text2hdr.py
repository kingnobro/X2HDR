import os
import torch
import numpy as np
import argparse
import logging
from typing import Tuple
from diffusers import FluxPipeline
from utils.util_hdr import (
    save_hdr_image,
    pu21_decode_rgb,
    scale_to_L_peak,
    get_luminance_percentile
)
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_pipeline(model_id: str, lora_path: str) -> FluxPipeline:
    """Initialize and setup the FLUX pipeline with LoRA weights."""
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    if not hasattr(pipe.text_encoder, "text_model"):
        # Kohya-converted LoRA keys retain a text_model. prefix that Transformers 5
        # removed from CLIP, so no layers match and rank inference raises IndexError.
        # Remove the extra prefix after conversion to match the current CLIP layers.
        state_dict, network_alphas = pipe.lora_state_dict(lora_path, return_alphas=True)
        if network_alphas:
            state_dict.update(network_alphas)
        old_prefix = "text_encoder.text_model."
        renamed_count = sum(key.startswith(old_prefix) for key in state_dict)
        if renamed_count:
            state_dict = {
                "text_encoder." + key[len(old_prefix):] if key.startswith(old_prefix) else key: value
                for key, value in state_dict.items()
            }
            logger.info("Normalized %d CLIP LoRA keys for Transformers 5", renamed_count)
        pipe.load_lora_weights(state_dict)
    else:
        pipe.load_lora_weights(lora_path)
    return pipe

def generate_latents(
    pipe: FluxPipeline,
    prompt: str,
    height: int,
    width: int,
    guidance_scale: float,
    num_steps: int,
    max_seq_len: int,
    seed: int
) -> torch.Tensor:
    """Generate image latents from the prompt."""
    return pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        max_sequence_length=max_seq_len,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent"
    ).images

def decode_latents(
    pipe: FluxPipeline,
    latents: torch.Tensor,
    height: int,
    width: int
) -> np.ndarray:
    """Decode latents to image tensor in range [-1, 1]."""
    # Unpack and decode latents
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    
    x = pipe.vae.decode(latents, return_dict=False)[0]
    
    # Post-process tensor: clamp and convert to [B, H, W, C] format
    x = x.detach().clamp(-1, 1).permute(0, 2, 3, 1)
    
    return x.float().cpu().numpy()[0]

def process_to_sdr(decoded_image: np.ndarray) -> Image.Image:
    """Convert decoded image tensor to 8-bit SDR PIL Image."""
    img_array = (127.5 * (decoded_image + 1.0)).astype(np.uint8)
    return Image.fromarray(img_array)

def process_to_hdr(decoded_image: np.ndarray, target_luminance: float) -> np.ndarray:
    """Process decoded image tensor to HDR format."""
    # Convert from [-1, 1] to [0, 1] PU21 space
    pu21_image = np.clip((decoded_image + 1.0) / 2.0, 0.0, 1.0)
    hdr_image = pu21_decode_rgb(pu21_image)
    hdr_image, _ = scale_to_L_peak(hdr_image)
    
    # Scale luminance
    highest_luminance = get_luminance_percentile(hdr_image, percentile=99.5)
    scale = min(target_luminance / highest_luminance, 1.0)
    return hdr_image * scale

def parse_prompt_with_options(prompt_line, default_width, default_height, default_steps, default_guidance, default_seed, default_target_luminance):
    """
    Parse a prompt line with options like:
    "prompt text --w 512 --h 512 --d 42 --g 3.5 --s 30"
    Returns: (prompt, seed, width, height, steps, guidance, target_luminance)
    """
    options = prompt_line.split("--")
    prompt = options[0].strip()
    
    # Set defaults
    width = default_width
    height = default_height
    steps = default_steps
    guidance = default_guidance
    seed = default_seed
    target_luminance = default_target_luminance
    
    for opt in options[1:]:
        try:
            opt = opt.strip()
            if opt.startswith("w"):
                width = int(opt[1:].strip())
            elif opt.startswith("h"):
                height = int(opt[1:].strip())
            elif opt.startswith("s"):
                steps = int(opt[1:].strip())
            elif opt.startswith("d"):
                seed = int(opt[1:].strip())
            elif opt.startswith("g"):
                guidance = float(opt[1:].strip())
        except ValueError as e:
            logger.error(f"Invalid option: {opt}, {e}")
    
    return prompt, seed, width, height, steps, guidance, target_luminance

def generate_single_image(
    pipe: FluxPipeline,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    max_sequence_length: int,
    target_luminance: float,
    output_dir: str,
    file_name: str = None
):
    """Generate a single image from prompt."""
    logger.info(f"Generating image for prompt: {prompt}")
    logger.info(f"Parameters: seed={seed}, size={width}x{height}, guidance={guidance_scale}, steps={num_inference_steps}")
    
    # Generate latents
    latents = generate_latents(
        pipe,
        prompt,
        height,
        width,
        guidance_scale,
        num_inference_steps,
        max_sequence_length,
        seed
    )
    
    # Decode latents
    decoded_image = decode_latents(
        pipe,
        latents,
        height,
        width
    )
    
    # Process to SDR
    sdr_image = process_to_sdr(decoded_image)
    
    # Process to HDR
    hdr_image = process_to_hdr(decoded_image, target_luminance)
    
    # Save images
    if file_name is None:
        sdr_path = os.path.join(output_dir, f"seed_{seed}.png")
        hdr_path = os.path.join(output_dir, f"seed_{seed}.exr")
    else:
        base_name = os.path.splitext(file_name)[0]
        sdr_path = os.path.join(output_dir, f"{base_name}.png")
        hdr_path = os.path.join(output_dir, f"{base_name}.exr")
    
    sdr_image.save(sdr_path)
    logger.info(f"Saved SDR image to {sdr_path}")
    
    save_hdr_image(hdr_image.astype(np.float32), hdr_path)
    logger.info(f"Saved HDR image to {hdr_path}")

def process_batch_prompts(batch_prompts_file, pipe, args):
    """Process prompts from a batch file."""
    if not os.path.isfile(batch_prompts_file):
        logger.error(f"Batch prompts file not found: {batch_prompts_file}")
        return
    
    with open(batch_prompts_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    
    if not prompts:
        logger.error("No valid prompts found in batch file")
        return
    
    logger.info(f"Found {len(prompts)} prompts to process")
    
    for i, prompt_line in enumerate(prompts, 1):
        logger.info(f"Processing prompt {i}/{len(prompts)}")
        
        # Parse prompt with options
        prompt, seed, width, height, steps, guidance, target_luminance = parse_prompt_with_options(
            prompt_line, 
            args.width, 
            args.height, 
            args.num_inference_steps, 
            args.guidance_scale, 
            args.seed,
            args.target_luminance
        )
        
        if not prompt:
            logger.warning(f"Empty prompt on line {i}, skipping")
            continue
        
        try:
            generate_single_image(
                pipe,
                prompt,
                seed,
                width,
                height,
                guidance,
                steps,
                args.max_sequence_length,
                target_luminance,
                args.output_dir,
                f"prompt_{i}"
            )
        except Exception as e:
            logger.error(f"Error generating image for prompt {i}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="FLUX Diffuser LoRA Test with batch processing support")
    parser.add_argument("--model_id", type=str, default="models/Flux", help="Path to the FLUX model")
    parser.add_argument("--lora_path", type=str, default="models/text2hdr_lora.safetensors", help="Path to LoRA weights")
    parser.add_argument("--prompt", type=str, default="PU21, masterpiece, 4K, sharp and detailed, high resolution, best quality, A grand, dimly lit hall with a single candle in the foreground", help="Single prompt to generate")
    parser.add_argument("--batch_prompts", type=str, default=None, help="File containing prompts with options for batch processing. If not provided, single prompt mode will be used.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--target_luminance", type=float, default=16.0, help="Target luminance for HDR")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup pipeline
    logger.info("Setting up FLUX pipeline...")
    pipe = setup_pipeline(args.model_id, args.lora_path)
    logger.info("Pipeline setup complete")
    
    if args.batch_prompts:
        # Process batch prompts from file
        logger.info(f"Processing batch prompts from {args.batch_prompts}")
        process_batch_prompts(args.batch_prompts, pipe, args)
    else:
        # Single prompt mode
        logger.info("Processing single prompt")
        generate_single_image(
            pipe,
            args.prompt,
            args.seed,
            args.width,
            args.height,
            args.guidance_scale,
            args.num_inference_steps,
            args.max_sequence_length,
            args.target_luminance,
            args.output_dir,
            "output_hdr"
        )
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
