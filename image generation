!nvidia-smi

!pip install --upgrade diffusers[torch]
!pip install transformers accelerate

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipeline.to("cuda")

import random
import os

os.makedirs('/content/faces/happy', exist_ok=True)
os.makedirs('/content/faces/sad', exist_ok=True)
os.makedirs('/content/faces/angry', exist_ok=True)
os.makedirs('/content/faces/surprised', exist_ok=True)

ethnicities = [
    'a latino', 'a white', 'a black',
    'a middle eastern', 'an indian', 'an asian'
]

genders = ['male', 'female']

emotion_prompts = {
    'happy': 'smiling',
    'sad': 'frowning, sad face expression, crying',
    'surprised': 'surprised, opened mouth, raised eyebrows',
    'angry': 'angry'
}


import random
import os
import matplotlib.pyplot as plt
import torch

camera_angles = [
    "front view",
    "slight left angle",
    "slight right angle"
]

lighting = [
    "soft natural lighting",
    "studio lighting",
    "dramatic lighting"
]

ages = [
    "young adult",
    "middle aged"
]

emotion_intensity = {
    "happy": ["slight smile", "big smile, joyful expression"],
    "sad": ["sad expression", "crying, teary eyes"],
    "angry": ["angry", "very angry, clenched jaw"],
    "surprised": ["surprised", "very surprised, wide eyes"]
}

for j in range(2):
    for emotion in emotion_prompts.keys():

        ethnicity = random.choice(ethnicities)
        gender = random.choice(genders)
        age = random.choice(ages)
        camera = random.choice(camera_angles)
        light = random.choice(lighting)
        intensity = random.choice(emotion_intensity[emotion])

        prompt = (
            f"Medium-shot portrait of a {age} {ethnicity} {gender}, "
            f"{intensity}, {camera}, looking at the camera, "
            f"{light}, color photography, photorealistic, "
            "realistic, crisp focus, depth of field, 50mm"
        )

        negative_prompt = (
            "3d, cartoon, anime, sketches, worst quality, low quality, "
            "lowres, monochrome, grayscale, plastic, fake, "
            "disfigured, deformed, blurry, bad anatomy, "
            "watermark, grainy, signature"
        )

        generator = torch.Generator(device="cuda").manual_seed(
            random.randint(0, 10_000)
        )

        img = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=27,
            generator=generator
        ).images[0]

        img.save(f"/content/faces/{emotion}/{str(j).zfill(4)}.png")

        plt.imshow(img)
        plt.axis("off")
        plt.show()


!zip -r faces.zip /content/faces
