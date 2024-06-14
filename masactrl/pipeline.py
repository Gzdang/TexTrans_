import torch

import numpy as np

from tqdm import tqdm
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from torchvision.utils import save_image


class MyPipeline(StableDiffusionXLPipeline):
    def upcast_vae(self):
        self.vae.to(dtype=torch.float32)

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

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(image, (Image.Image, PngImagePlugin.PngImageFile)):
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)["sample"]

        return image  # range [-1, 1]

    def add_control(self, controlnet):
        self.controlnet = controlnet

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        base_resolution=512,
        is_combine=False,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        control={},
        control_scale=1,
        **kwds,
    ):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeds
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(prompt)

        # image size
        if is_combine:
            width = base_resolution * 2
            height = base_resolution * 3
        else:
            width = base_resolution
            height = base_resolution

        # add embeds
        add_text_embeds = pooled_prompt_embeds

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(DEVICE)
        add_text_embeds = add_text_embeds.to(DEVICE)
        add_time_ids = add_time_ids.to(DEVICE)

        # controlnet preprocess
        if control.get("depth") is not None:
            depth = control["depth"]
            control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            if isinstance(depth, list):
                source_depth = control_image_processor.preprocess(depth[0], height=height, width=width)
                tar_depth = control_image_processor.preprocess(depth[1], height=height, width=width)
                depth = torch.cat([source_depth, tar_depth] * (prompt_embeds.shape[0] // 2)).to(
                    self.depth_controlnet.device
                )
            else:
                depth = control_image_processor.preprocess(depth, height=height, width=width)
                depth = depth.expand(prompt_embeds.shape[0], -1, -1, -1).to(self.depth_controlnet.device)
            control["depth"] = depth.to(self.depth_controlnet.dtype)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)

                # if 900<t:
                #     latents_cur = feat_adain(latents_cur, latents_ref)
                latents = torch.cat([latents_ref, latents_cur])

            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            # predict the noise
            down_block_res_samples = None
            mid_block_res_sample = None
            if control.get("depth") is not None:
                depth = control["depth"]
                down_block_depth, mid_block_depth = self.depth_controlnet(
                    model_inputs,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=depth,
                    conditioning_scale=control_scale,
                    added_cond_kwargs=added_cond_kwargs,
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

            noise_pred = self.unet(
                model_inputs,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred[1:2], noise_pred[-1:]
                # noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred.repeat(2, 1, 1, 1), t, latents)
            # latents, pred_x0 = self.step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents.to(torch.float32), return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        control={},
        control_scale=1,
        base_resolution=512,
        is_combine=False,
        return_intermediates=False,
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

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(prompt)

        # define initial latents
        image = self.image_processor.preprocess(image).cuda()
        latents = self.image2latent(image).to(self.unet.dtype)
        start_latents = latents

        if control.get("depth") is not None:
            depth = control["depth"]
            control_image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
            )
            if isinstance(depth, list):
                source_depth = control_image_processor.preprocess(depth[0], height=height, width=width)
                tar_depth = control_image_processor.preprocess(depth[1], height=height, width=width)
                depth = torch.cat([source_depth, tar_depth] * (batch_size // 2)).to(self.depth_controlnet.device)
            else:
                depth = control_image_processor.preprocess(depth, height=height, width=width)
                depth = depth.expand(batch_size, -1, -1, -1).to(self.depth_controlnet.device)
            control["depth"] = depth.to(self.depth_controlnet.dtype)

        add_text_embeds = pooled_prompt_embeds

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(DEVICE)
        add_text_embeds = add_text_embeds.to(DEVICE)
        add_time_ids = add_time_ids.to(DEVICE)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # model_inputs = self.scheduler.scale_model_input(model_inputs, t)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            down_block_res_samples = None
            mid_block_res_sample = None
            if control.get("depth") is not None:
                depth = control["depth"]
                down_block_depth, mid_block_depth = self.depth_controlnet(
                    model_inputs,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=depth,
                    conditioning_scale=control_scale,
                    added_cond_kwargs=added_cond_kwargs,
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
                # down_block_res_samples = [torch.stack([torch.zeros_like(t[0]), t[1], torch.zeros_like(t[2]), t[3]]) for t in down_block_res_samples]
                # mid_block_res_sample = torch.stack([torch.zeros_like(mid_block_res_sample[0]), mid_block_res_sample[1], torch.zeros_like(mid_block_res_sample[2]), mid_block_res_sample[3]])

            noise_pred = self.unet(
                model_inputs,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

            # save_image(self.latent2image(pred_x0.to(torch.float32), "pt"), "test.png")

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
