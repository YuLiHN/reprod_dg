import torch
import os
from training.classifier import EncoderUNetModel

def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )

def get_discriminator(latent_extractor_ckpt, discriminator_ckpt, condition, img_resolution=32, device='cuda', enable_grad=True):
    classifier = load_classifier(latent_extractor_ckpt, img_resolution, device)
    discriminator = load_discriminator(discriminator_ckpt, device)
    def evaluate(perturbed_inputs, timesteps=None, condition=None):
        with torch.enable_grad() if enable_grad else torch.no_grad():
            adm_features = classifier(perturbed_inputs, timesteps=timesteps, feature=True)
            prediction = discriminator(adm_features, timesteps, sigmoid=True, condition=condition).view(-1)
        return prediction
    return evaluate

def load_classifier(ckpt_path, img_resolution, device):
    classifier_args = dict(
      image_size=img_resolution,
      classifier_use_fp16=False,
      classifier_width=128,
      classifier_depth=4 if img_resolution in [64, 32] else 2,
      classifier_attention_resolutions="32,16,8",
      classifier_use_scale_shift_norm=True,
      classifier_resblock_updown=True,
      classifier_pool="attention",
      in_channels=3,
      out_channels=1000,
    )
    classifier = create_classifier(**classifier_args)
    classifier.to(device)
    if ckpt_path is not None:
        ckpt_path = os.getcwd() + ckpt_path
        classifier_state = torch.load(ckpt_path, map_location="cpu")
        classifier.load_state_dict(classifier_state)
    classifier.eval()
    return classifier

def load_discriminator(ckpt_path, device, eval=False, channel=512):
    discriminator_args = dict(
      image_size=8,
      classifier_use_fp16=False,
      classifier_width=128,
      classifier_depth=2,
      classifier_attention_resolutions="32,16,8",
      classifier_use_scale_shift_norm=True,
      classifier_resblock_updown=True,
      classifier_pool="attention",
      out_channels=1,
      in_channels=channel,
    )
    discriminator = create_classifier(**discriminator_args)
    discriminator.to(device)
    if ckpt_path is not None:
        ckpt_path = os.getcwd() + ckpt_path
        discriminator_state = torch.load(ckpt_path, map_location="cpu")
        discriminator.load_state_dict(discriminator_state)
    if eval:
        discriminator.eval()
    return discriminator

def create_classifier(
    image_size,
    in_channels,
    out_channels,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 4)
    elif image_size == 16:
        channel_mult = (1, 2)
    elif image_size == 8:
        channel_mult = (1,)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=classifier_width,
        out_channels=out_channels,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def get_grad_log_ratio(discriminator, unnormalized_input, std_wve_t, img_resolution,time_min, time_max, class_labels):
    mean_vp_tau, tau = transform_unnormalized_wve_to_normalized_vp(std_wve_t) ## VP pretrained classifier
    if tau.min() > time_max or tau.min() < time_min or discriminator == None:
        return torch.zeros_like(unnormalized_input), 10000000. * torch.ones(unnormalized_input.shape[0], device=unnormalized_input.device)
    # import pdb;pdb.set_trace()

    # input = mean_vp_tau[:,None,None,None] * unnormalized_input
    input = mean_vp_tau * unnormalized_input
    with torch.enable_grad():
        x_ = input.float().clone().detach().requires_grad_()
        # if img_resolution == 64: # ADM trained UNet classifier for 64x64 with Cosine VPSDE
        #     tau = vpsde.compute_t_cos_from_t_lin(tau)
        tau = torch.ones(input.shape[0], device=tau.device) * tau
        log_ratio = get_log_ratio(discriminator, x_, tau, class_labels)
        discriminator_guidance_score = torch.autograd.grad(outputs=log_ratio.sum(), inputs=x_, retain_graph=False)[0]
        # print(mean_vp_tau.shape)
        # print(std_wve_t.shape)
        # print(discriminator_guidance_score.shape)
        # discriminator_guidance_score *= - ((std_wve_t[:,None,None,None] ** 2) * mean_vp_tau[:,None,None,None])
        discriminator_guidance_score *= - ((std_wve_t ** 2) * mean_vp_tau)
    # if log:
    #   return discriminator_guidance_score, log_ratio
    return discriminator_guidance_score, log_ratio

def get_log_ratio(discriminator, input, time, class_labels):
    if discriminator == None:
        return torch.zeros(input.shape[0], device=input.device)
    else:
        logits = discriminator(input, timesteps=time, condition=class_labels)
        prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
        log_ratio = torch.log(prediction / (1. - prediction))
        return log_ratio

''' 
the following implementations is for in supp. mat. equation 8,
the basic idea is to bridge the different beta strategies of EDM/ADM
'''
def compute_tau(std_wve_t, beta_min, beta_max):
    tau = -beta_min + torch.sqrt(beta_min ** 2 + 2. * (beta_max - beta_min) * torch.log(1. + std_wve_t ** 2))
    tau /= beta_max - beta_min
    return tau

def marginal_prob(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def transform_unnormalized_wve_to_normalized_vp(t, beta_min=0.1, beta_max=20., std_out=False):
    tau = compute_tau(t, beta_min, beta_max)
    mean_vp_tau, std_vp_tau = marginal_prob(tau, beta_min, beta_max)
    if std_out:
        return mean_vp_tau, std_vp_tau, tau
    return mean_vp_tau, tau

