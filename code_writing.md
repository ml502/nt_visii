1.prompt 使用visii中学习到的prompt
learned_prompt = optimize_prompt(model, preprocess, args1, device, target_images=[target_image])

2.保留了image_guidance_scale
  if do_classifier_free_guidance:
      noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
      noise_pred = (
          noise_pred_uncond
          + guidance_scale * (noise_pred_text - noise_pred_image)
          + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
      )
