import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterXL
from numpy.random import randint

base_model_path = "/ML-A100/team/mm/wangtao/share/models/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# read image prompt
image_paths=["/ML-A100/team/mm/shuyu/IP-Adapter-fox/assets/images/woman.png","/ML-A100/team/mm/shuyu/IP-Adapter-fox/assets/images/woman1.jpg"]
input_images=[Image.open(path) for path in image_paths]
for i,image in enumerate(input_images):
    if image.mode=='RGBA':
        input_images[i]=image.convert('RGB')
    input_images[i]=image.resize((512,512))


# multimodal prompts
num_samples = 3
random_seed = randint(0, 2**32 - 1)
images = ip_model.generate(pil_image=input_images, num_samples=num_samples, num_inference_steps=30, seed=random_seed,
        prompt="A girl looks back", scale=0.6)
print(len(images))


# save results
num_input=len(input_images)
num_images=len(images)

if num_images>0:
    width,height=images[0].size
    total_width=width*num_images
    
    new_image=Image.new('RGB',(total_width,height))

    x_offset=0
    for img in images:
        new_image.paste(img,(x_offset,0))
        x_offset += width

from datetime import datetime
now=datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S") 
new_image.save(f"/ML-A100/team/mm/shuyu/IP-Adapter-fox/output_images/{timestamp}.png")




