from flask import Flask, request, send_file
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)

import io
from PIL import Image

app = Flask(__name__)

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Corcelio/mobius",
    vae=vae,
    torch_dtype=torch.float16
)
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json['prompt']
    guidance_scale = request.json.get('guidance_scale', 7)
    clip_skip = request.json.get('clip_skip', -3)
    num_inference_steps = request.json.get('num_inference_steps', 50)

    prompt = f"best quality, HD, *aesthetic* {prompt}"

    image = pipe(
        prompt,
        negative_prompt="",
        width=1024,
        height=1024,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        clip_skip=clip_skip
    ).images[0]

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)