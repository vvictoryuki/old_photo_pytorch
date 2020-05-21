import torch
# from torch import nn


def latent_loss(mu, logvar):
    mu_sq = mu * mu
    std = torch.exp(logvar)
    std_sq = std * std
    return 0.5 * torch.mean(mu_sq + std_sq - torch.log(std_sq) - 1)


def VAE_reconstruct_loss(outputs, inputs):
    # return nn.MSELoss()(outputs, inputs)
    return torch.nn.L1Loss()(outputs, inputs)


def LSLoss_gen(sy_):
    return 0.5 * torch.mean(torch.pow((sy_-1), 2))


def LSLoss_disc_real(sy):
    return 0.5 * torch.mean(torch.pow((sy-1), 2))


def LSLoss_disc_fake(sy_):
    return 0.5 * torch.mean(torch.pow(sy_, 2))


def GAN_latent_loss_gen(szx):
    return torch.mean(torch.pow((szx-1), 2))


def GAN_latent_loss_disc_fake(szx):
    return torch.mean(torch.pow(szx, 2))


def GAN_latent_loss_disc_real(szr):
    return torch.mean(torch.pow((szr-1), 2))


def latent_space_loss(zx_, zy):
    return torch.nn.L1Loss()(zx_, zy)


def feature_map_l1loss(fm1, fm2):
    return torch.nn.L1Loss()(fm1, fm2)


def feature_matching_loss(fm_x_d, fm_y_d, fm_x_vgg, fm_y_vgg):
    d = []
    vgg = []

    for idx in range(len(fm_x_d)):
        d.append(feature_map_l1loss(fm_x_d[idx], fm_y_d[idx]))
    for idx in range(len(fm_x_vgg)):
        vgg.append(feature_map_l1loss(fm_x_vgg[idx], fm_y_vgg[idx]))

    return d[0] + d[1] + d[2] + d[3] + d[4] + vgg[0] + vgg[1] + vgg[2] + vgg[3] + vgg[4]
