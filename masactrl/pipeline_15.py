import torch

import numpy as np

from tqdm import tqdm
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from torchvision.utils import save_image
from torchvision.transforms.v2 import Resize


class MyPipeline(StableDiffusionPipeline):
    def next_step(self, model_output: torch.FloatTensor, timestep: int, x: torch.FloatTensor, eta=0.0, verbose=False):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float = 0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0
    
    def prev_step(self, x_t, x_0, timestep):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        noise_pred = (x_t - alpha_prod_t**0.5 * x_0) / beta_prod_t**0.5

        return alpha_prod_t_prev**0.5 * x_0 + (1 - alpha_prod_t_prev) ** 0.5 * noise_pred

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(image, (Image.Image, PngImagePlugin.PngImageFile)):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            if image.shape[1] == 4:
                image = image[:, :-1, :, :]
        # input image density range [-1, 1]
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents.half()
    
    @torch.no_grad()
    def latent2image_nograd(self, latents, return_type="np"):
        return self.latent2image(latents, return_type)
        
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            # image = (image / 2 + 0.5).clamp(0, 1)
            image = (image / 2 + 0.5)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)["sample"]

        return image  # range [-1, 1]

    def add_control(self, controlnet):
        self.controlnet = controlnet

    # @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        base_resolution=512,
        is_combine=False,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        control={},
        control_scale=1,
        uv_model=None,
        **kwds,
    ):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        if is_combine:
            width = base_resolution * 2
            height = base_resolution * 3
        else:
            width = base_resolution
            height = base_resolution

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert (
                latents.shape == latents_shape
            ), f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        if control.get("depth") is not None:
            depth = control["depth"]
            control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            if isinstance(depth, list):
                source_depth = control_image_processor.preprocess(depth[0], height=height, width=width)
                tar_depth = control_image_processor.preprocess(depth[1], height=height, width=width)
                depth = torch.cat([source_depth, tar_depth] * (text_embeddings.shape[0] // 2)).to(
                    self.controlnet.device
                )
            else:
                depth = control_image_processor.preprocess(depth, height=height, width=width)
                depth = depth.expand(text_embeddings.shape[0], -1, -1, -1).to(self.controlnet.device)
            control["depth"] = depth

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)

                mean_ref = torch.mean(latents_ref, dim=(2,3), keepdim=True)
                std_ref = torch.std(latents_ref, dim=(2,3), keepdim=True)
                mean_tar = torch.mean(latents_cur, dim=(2,3), keepdim=True)
                std_tar = torch.std(latents_cur, dim=(2,3), keepdim=True)

                latents_cur = ((latents_cur - mean_tar)/std_tar)*std_ref + mean_ref

                # if 900<t:
                #     latents_cur = feat_adain(latents_cur, latents_ref)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])

            @torch.no_grad()
            def get_control():
                down_block_res_samples = None
                mid_block_res_sample = None
                if control.get("depth") is not None:
                    depth = control["depth"]
                    down_block_depth, mid_block_depth = self.controlnet(
                        model_inputs,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=depth.half(),
                        conditioning_scale=control_scale,
                        return_dict=False,
                    )
                    if down_block_res_samples is None and mid_block_res_sample is None:
                        down_block_res_samples = down_block_depth
                        mid_block_res_sample = mid_block_depth
                    else:
                        down_block_res_samples = [
                            samples_prev + samples_curr
                            for samples_prev, samples_curr in zip(down_block_res_samples, down_block_depth)
                        ]
                        mid_block_res_sample += mid_block_depth
                return down_block_res_samples, mid_block_res_sample
            down_block_res_samples, mid_block_res_sample = get_control()

            # down_block_res_samples = [torch.stack([torch.zeros_like(t[0]), t[1], t[2], t[3]]) for t in down_block_res_samples]
            # mid_block_res_sample = torch.stack([torch.zeros_like(mid_block_res_sample[0]), mid_block_res_sample[1], mid_block_res_sample[2], mid_block_res_sample[3]])

            @torch.no_grad()
            def get_noise():
                return self.unet(
                    model_inputs,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            noise_pred = get_noise()

            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred[1:2], noise_pred[-1:]
                # noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            # latents, pred_x0 = self.step(noise_pred.repeat(2,1,1,1), t, latents)

            # TODO
            # 把latent转回图片
            # uv project回模型
            # 模型重新渲染得到图片
            # encode 回 latent
            # latent 插值
            latents_, pred_x0 = self.step(noise_pred, t, latents)
            # if uv_model is not None:
            if uv_model is not None and i >= 25:
                mask = (depth[1:] != 0).int()
                
                # 预测x0 （0-1）
                image = self.latent2image(pred_x0[1:].float(), return_type="pt").detach()
                target = mask * image
                save_image(target, "./output/proj/image_.png")

                # 显示控制
                res = uv_model.project(target)
                # res = image
                save_image(res, "./output/proj/image_res.png")

                # z = pred_x0[:1].float().clone().detach()
                # z.requires_grad_(True)
                # optim = torch.optim.Adam([z], 0.01)
                # for _ in range(200):
                #     out = self.latent2image(z, "pt")
                #     loss = 10 * torch.nn.functional.l1_loss(out, res.detach())
                #     loss.backward()
                #     optim.step()
                #     optim.zero_grad()
                # save_image(self.latent2image(z, "pt"), "./output/proj/image.png")
                
                # latent = self.prev_step(latents[-1:], res, t)
                # TODO 这里有问题
                # latent = self.prev_step(latents[-1:], self.image2latent((res-1)*2), t)
                latent = self.prev_step(latents[-1:], self.image2latent(res*2-1), t)
                latents = torch.cat([latent]*2).half()
            else:
                latents = latents_

                # forward = torch.cat([z.detach().half()]*2)
                # for t_ in reversed(self.scheduler.timesteps[i+1:]):
                #     @torch.no_grad()
                #     def back():
                #         return self.unet(forward, t_, encoder_hidden_states=text_embeddings).sample
                #     pred = back()
                #     forward, _ = self.next_step(pred, t_, forward)

                # latents = torch.cat([latents[:1], forward])

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents[-1:].float(), return_type="pt")
        if uv_model is not None:
            save_image(uv_model.texture_map, "./output/proj/texture.png")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    # @torch.no_grad()
    def invert(
        self,
        image,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        control={},
        control_scale=1,
        style_image=None,
        base_resolution=512,
        is_combine=False,
        **kwds,
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = 1
        if isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        if is_combine:
            width = base_resolution * 2
            height = base_resolution * 3
        else:
            width = base_resolution
            height = base_resolution

        if style_image is not None:
            style_image = self.feature_extractor(style_image, return_tensors="pt").pixel_values
            text_embeddings = self.image_encoder(style_image.to(self.image_encoder.device)).image_embeds
            text_embeddings = text_embeddings.unsqueeze(1)

        else:
            # text embeddings
            text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

        # define initial latents
        latents = self.image2latent(image)

        if guidance_scale > 1.0:
            if style_image is not None:
                unconditional_embeddings = torch.zeros_like(text_embeddings)
            else:
                unconditional_input = self.tokenizer(
                    [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
                )
                unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        if control.get("depth") is not None:
            depth = control["depth"]
            control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            if isinstance(depth, list):
                source_depth = control_image_processor.preprocess(depth[0], height=height, width=width)
                tar_depth = control_image_processor.preprocess(depth[1], height=height, width=width)
                depth = torch.cat([source_depth, tar_depth] * (text_embeddings.shape[0] // 2)).to(
                    self.controlnet.device
                )
            else:
                depth = control_image_processor.preprocess(depth, height=height, width=width)
                depth = depth.expand(text_embeddings.shape[0], -1, -1, -1).to(self.controlnet.device)
            control["depth"] = depth

        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)

        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            @torch.no_grad()
            def get_control():
                down_block_res_samples = None
                mid_block_res_sample = None
                if control.get("depth") is not None:
                    depth = control["depth"]
                    down_block_depth, mid_block_depth = self.controlnet(
                        model_inputs,
                        t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=depth.half(),
                        conditioning_scale=control_scale,
                        return_dict=False,
                    )
                    if down_block_res_samples is None and mid_block_res_sample is None:
                        down_block_res_samples = down_block_depth
                        mid_block_res_sample = mid_block_depth
                    else:
                        down_block_res_samples = [
                            samples_prev + samples_curr
                            for samples_prev, samples_curr in zip(down_block_res_samples, down_block_depth)
                        ]
                        mid_block_res_sample += mid_block_depth
                return down_block_res_samples, mid_block_res_sample
            down_block_res_samples, mid_block_res_sample = get_control()

            @torch.no_grad()
            def get_noise():
                return self.unet(
                    model_inputs,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            noise_pred = get_noise()

            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
            # save_image(self.latent2image(pred_x0, "pt"), "test.png")

        return latents, latents_list

    def add_noise(self, img, noise):
        latents = self.image2latent(img)
        self.scheduler.set_timesteps(30)
        return self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])
