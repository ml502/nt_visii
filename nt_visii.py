import argparse
import os
import yaml
# 指定gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
from visii import StableDiffusionVisii,preprocess

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_folder', type=str, default='ip2p_painting1_0_0.png')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--image_folder', type=str, default='./images')
    parser.add_argument('--config_file', type=str, default='configs/config_ip2p.yaml')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--guidance_scale', type=int, default=8)
    parser.add_argument('--prompt', type=str, default='a husky')
    parser.add_argument('--hybrid_ins', type=bool, default=False)

    parser.add_argument('--subfolder', type=str, default=None)
    parser.add_argument('--init_expname', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if args.subfolder is not None:
        config['data']['subfolder'] = args.subfolder
    if args.init_expname is not None:
        config['exp']['init_expname'] = args.init_expname
    if args.image_folder is not None:
        config['data']['image_folder'] = args.image_folder

    model_id = config['model']['model_id'] #"/data/ml/instruct-pix2pix"
    pipe = StableDiffusionVisii.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    subfolders = os.listdir(config['data']['image_folder']) #"./images/"
    subfolders = [x for x in subfolders if config['data']['subfolder'] in x] #"painting1"

    for folder in subfolders:
        current_folder = os.path.join(config['data']['image_folder'], folder)
        print('Learn with:',current_folder) #Learn with: ./images/painting1

        # 加载图像对进行学习 (0_0.png and 0_1.png)
        before_path = os.path.join(current_folder, '0_0.png')
        before_image = Image.open(before_path).resize((512, 512)).convert('RGB') #0_0.png
        after_image = Image.open(before_path.replace('_0.', '_1.')).resize((512, 512)).convert('RGB') #0_1.png

        log_dir = os.path.join(args.log_dir, 'nt_logs') #'logs/nt_logs'
        os.makedirs(log_dir, exist_ok=True)
        
        if config['exp']['prompt_type'] == 'hard':
            prompt = config['exp']['init_prompt']
            print('Initialize with hard prompt: ', prompt)
        elif config['exp']['prompt_type'] == 'learn':
            config_path = "./configs/hard_prompts_made_easy.json"
            from pez import *
            print("Finding initial caption...")
            args1 = argparse.Namespace()
            args1.__dict__.update(read_json(config_path))
            args1.print_new_best = False

            # load CLIP model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(args1.clip_model, pretrained=args1.clip_pretrain, device=device)
            #"clip_model": "ViT-H-14","clip_pretrain": "/ViT-H-14/open_clip_pytorch_model.bin"
            print(f"Running for {args1.iter} steps.") #Running for 3000 steps.

            learned_prompt = optimize_prompt(model, preprocess, args1, device, target_images=[after_image])
            print(learned_prompt)
            prompt = learned_prompt
            del model, preprocess, args1, learned_prompt
        else:
            print("What is your prompt?")
            exit()

        with open(os.path.join(log_dir, "learned_prompt.txt"), "w") as text_file:
            text_file.write("{}".format(prompt))
        
        print('Save learned prompt to: ', os.path.join(log_dir, "learned_prompt.txt"))
        print('Init prompt: ', prompt)

        print('Learning editing direction')
        before_noise_pred_text, before_noise_pred_image, before_noise_pred_uncond = pipe.get_specific_noise(prompt,before_image) #torch.Size([1, 4, 64, 64])
        after_noise_pred_text, after_noise_pred_image, after_noise_pred_uncond = pipe.get_specific_noise(prompt,after_image)
        c_t = after_noise_pred_text - before_noise_pred_uncond

        #修改c_t
        c_t = c_t.view(c_t.shape[0],-1) #[1,4*64*64]
        c_t = torch.matmul(c_t,torch.randn(4*64*64,77*768).cuda())
        c_t = c_t.view(c_t.shape[0],77,768)
        uncond_tokens = [""]
        max_length = c_t.shape[1]
        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        negative_prompt_embeds = pipe.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipe.text_encoder.dtype, device=device)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # pix2pix has two  negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
        c_t = torch.cat([c_t, negative_prompt_embeds, negative_prompt_embeds])

        print('Start editing')
        img_path = './images/1_0.png'
        test_image = Image.open(img_path).convert("RGB").resize((512, 512))
        if args.hybrid_ins:
            with open(os.path.join(log_dir, 'learned_prompt.txt')) as f:
                init_prompt = f.read()
            
            target_image = pipe.test_concatenate(prompt_embeds=c_t,
                image=before_image,
                image_guidance_scale=1.5,
                guidance_scale=args.guidance_scale,
                num_inference_steps=20,
                prompt = args.prompt,
                init_prompt = init_prompt,
                num_images_per_prompt=1,
                ).images
        else:
            target_image = pipe.test(prompt_embeds=c_t,
                image=before_image,
                image_guidance_scale=1.5,
                guidance_scale=args.guidance_scale,
                num_inference_steps=20,
                num_images_per_prompt=1,
                ).images[0]
            
        location = os.path.join('results', 'nt_visii')
        os.makedirs(location, exist_ok=True)
        save_path = os.path.join(location, 'target_image.png')
        target_image.save(save_path)
        
        
        
        
