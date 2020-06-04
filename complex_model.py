import torch
import os
from torch import nn
from spectral_norm import SpectralNorm
import numpy as np
from dataset import YDS
from torch.utils.data import DataLoader
import imageio
import torch.nn.functional as F
import torchvision
# import cv2


class ResBlock(nn.Module):
    def __init__(self, input_nc, hidden_nc, output_nc, sample_type='none', sample_size=2):
        super(ResBlock, self).__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(input_nc, hidden_nc, 3, 1, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(hidden_nc, output_nc, 3, 1, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(input_nc, output_nc, 1, 1, 0))
        self.pooling = nn.AvgPool2d(2, 2) if sample_type == 'down' else None

        self.mianpass = nn.Sequential(
            self.nonlinearity, self.conv1, self.nonlinearity, self.conv2,
        )

        self.bypass = nn.Sequential(
            self.conv3,
        )

    def forward(self, x):
        if self.pooling:
            out = self.pooling(self.mianpass(x)) + self.pooling(self.bypass(x))
            return out
        else:
            out = self.mianpass(x) + self.bypass(x)
            return out


class ResBlockEncoder(nn.Module):
    def __init__(self, input_nc, hidden_nc, output_nc):
        super(ResBlockEncoder, self).__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(input_nc, hidden_nc, 3, 1, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(hidden_nc, output_nc, 3, 1, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(input_nc, output_nc, 1, 1, 0))

        self.mianpass = nn.Sequential(
            self.conv1, self.nonlinearity, self.conv2, self.nonlinearity, nn.AvgPool2d(2, 2)
        )

        self.bypass = nn.Sequential(
            nn.AvgPool2d(2, 2), self.conv3
        )

    def forward(self, x):
        # print(x.size())
        out = self.mianpass(x) + self.bypass(x)
        return out


class ResBlockDecoder(nn.Module):
    def __init__(self, input_nc, hidden_nc, output_nc):
        super(ResBlockDecoder, self).__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.conv1 = SpectralNorm(nn.Conv2d(input_nc, hidden_nc, 3, 1, 1))
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.conv3 = SpectralNorm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.mainpass = nn.Sequential(
            self.nonlinearity, self.conv1, self.nonlinearity, self.conv2
        )

        self.bypass = nn.Sequential(
            self.conv3
        )

    def forward(self, x):
        out = self.mainpass(x) + self.bypass(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = 5
        self.bcn = 64  # basic channel number
        self.mcn = 1024  # max channel number
        self.zcn = 128  # latent variable channel number

        self.baseEncoder = ResBlockEncoder(3, self.bcn, self.bcn)

        mult = 1
        for i in range(self.layer):
            pre_mult = mult
            mult = min(2**(i+1), self.mcn // self.bcn)
            block = ResBlock(self.bcn*pre_mult, self.bcn*mult, self.bcn*mult, 'down')
            setattr(self, "encoder" + str(i), block)

        self.inference = ResBlock(self.bcn*mult, self.bcn*mult, self.zcn*2, 'none')

    def reparameterize(self, fm):
        self.mu, self.std = torch.split(fm, self.zcn, dim=1)
        self.std = F.softplus(self.std)
        self.distribution = torch.distributions.Normal(self.mu, self.std)
        std_normal = torch.distributions.Normal(torch.zeros_like(self.mu), torch.ones_like(self.std))
        self.kl_divergence = torch.distributions.kl_divergence(std_normal, self.distribution)
        return self.distribution.rsample()

    def forward(self, x):
        feature = []
        x = self.baseEncoder(x)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            x = model(x)
            feature.append(x)
        h = self.inference(x)
        out = self.reparameterize(h)
        # for i in range(len(feature)):
        #     print(i, feature[i].size())
        return out, feature


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer = 6
        self.bcn = 64  # basic channel number
        self.mcn = 1024  # max channel number
        self.zcn = 128  # latent variable channel number
        self.nonlinearity = nn.LeakyReLU()

        mult = min(2**(self.layer-1), self.mcn // self.bcn)

        self.baseDecoder = nn.Sequential(
            ResBlock(self.zcn, self.bcn*mult, self.bcn*mult, 'none'),
            ResBlock(self.bcn*mult, self.bcn*mult, self.bcn*mult, 'none')
        )

        for i in range(self.layer):
            pre_mult = mult
            mult = min(2**(self.layer-i-1), self.mcn // self.bcn)
            block = ResBlockDecoder(self.bcn*pre_mult, self.bcn*mult, self.bcn*mult)
            setattr(self, "decoder" + str(i), block)

        self.output = nn.Sequential(
            self.nonlinearity,
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(self.bcn*mult, 3, 3)),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.baseDecoder(x)
        for i in range(self.layer):
            model = getattr(self, "decoder" + str(i))
            out = model(out)
        out = self.output(out)
        return out


class DiscriminatorLatent(nn.Module):
    def __init__(self):
        super(DiscriminatorLatent, self).__init__()
        self.layer = 5
        self.mcn = 1024

        mult = 1
        for i in range(self.layer):
            pre_mult = mult
            mult = min(4**(i+1), self.mcn // 1)
            block = ResBlock(self.mcn // pre_mult, self.mcn // mult, self.mcn // mult, 'none')
            setattr(self, "disc" + str(i), block)

        self.output = nn.Sequential(
            self.nonlinearity,
            SpectralNorm(nn.Conv2d(self.mcn, 1, 3))
        )

    def forward(self, x):
        for i in range(self.layer):
            model = getattr(self, "disc" + str(i))
            x = model(x)
        out = self.output(x)
        return out


class DiscriminatorImage(nn.Module):
    def __init__(self):
        super(DiscriminatorImage, self).__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.mcn = 1024

        self.encoder = Encoder()
        self.output = nn.Sequential(
            self.nonlinearity,
            SpectralNorm(nn.Conv2d(self.mcn, 1, 3))
        )

    def forward(self, x):
        _, feature = self.encoder(x)
        h = feature[-1]
        # print(h.size())
        out = self.output(h)
        return out, feature


def save_img(tensor_data, save_path):
    numpy_img = tensor_data.cpu().float().numpy()
    # numpy_img = (np.transpose(numpy_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    numpy_img = np.transpose(numpy_img, (1, 2, 0)) * 255.0
    numpy_img = numpy_img.astype(np.uint8)
    imageio.imwrite(save_path, numpy_img)


if __name__ == "__main__":
    e = Encoder()
    g = Generator()
    epoch = 50
    dict_path = os.path.join("./checkpoint", "y_training_epc"+str(epoch)+".pth")
    load_dict = torch.load(dict_path, map_location=torch.device('cpu')) if os.path.exists(dict_path) else None
    if load_dict:
        e.load_state_dict(load_dict['ey'])
        g.load_state_dict(load_dict['gy'])

    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    yds = YDS("./data/test", transform=transforms)
    data_loader = DataLoader(yds, batch_size=128, shuffle=True)
    for i, y_data in enumerate(data_loader):
        # y_data = y_data*2-1
        zy, _ = e(y_data)
        y_ = g(zy)
        for i in range(y_.size(0)):
            img_name = "test_"+str(i)+".png"
            save_img(y_[i].data, os.path.join("./data/out", img_name))

    # d = DiscriminatorImage()
    # t = torch.Tensor(3, 3, 256, 256)
    # zt, ft = e(t)
    # t_ = g(zt)
    # st_, ft_ = d(t_)
    # print("t", t.size())
    # print("zt", zt.size())
    # print("t_", t_.size())
    # print("st_", st_.size())
    # for i in range(len(ft)):
    #     print("f%d in ft:" % (i+1), ft[i].size())
