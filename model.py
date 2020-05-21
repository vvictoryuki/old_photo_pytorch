import torch
from torch import nn
# import torchvision.models as models


class ResBlock(nn.Module):
    def __init__(self, channel=64):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.residual_function = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel, self.channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel)
        )

    def forward(self, x):
        out = self.residual_function(x)
        out = out + x
        return nn.ReLU(inplace=True)(out)


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU()
        )

        self.resBlocks = nn.Sequential(
            ResBlock(), ResBlock(), ResBlock(), ResBlock()
        )

        self.fc_mu = nn.Linear(64*64*64, z_dim)
        self.fc_logvar = nn.Linear(64*64*64, z_dim)

    def reparameterize(self, h):
        self.mu = self.fc_mu(h)
        self.logvar = self.fc_logvar(h)
        sigma = torch.randn(self.mu.size())
        z = self.mu + sigma * self.logvar
        return z

    def forward(self, x):
        x = self.encoder_backbone(x)
        h = self.resBlocks(x)
        h = h.view(h.size(0), -1)  # flatten
        z = self.reparameterize(h)
        return z


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, 64*64*64)

        self.resBlocks = nn.Sequential(
            ResBlock(), ResBlock(), ResBlock(), ResBlock()
        )

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.ReLU()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, 64, 64)  # unflatten
        x = self.resBlocks(x)
        image = self.generator(x)
        return image


class PartialNonLocal(nn.Module):
    def __init__(self, channel=512):
        super(PartialNonLocal, self).__init__()
        self.channel = channel
        self.after_nonlocal = nn.Sequential(
            ResBlock(self.channel), ResBlock(self.channel)
        )

    def theta(self, x):
        x = nn.Conv2d(512, 256, 1, 1, 0)(x)
        x = x.view(x.size(0), 256, -1)
        return x

    def phi(self, x):
        x = nn.Conv2d(512, 256, 1, 1, 0)(x)
        x = x.view(x.size(0), 256, -1)
        return x

    def mu(self, x):
        x = nn.Conv2d(512, 256, 1, 1, 0)(x)
        x = x.view(x.size(0), 256, -1)
        return x

    def nu(self, x):
        x = x.view(x.size(0), 256, 64, 64)
        x = nn.Conv2d(256, 512, 1, 1, 0)(x)
        return x

    def mask_normalization(self, x, mask):
        """
        mask : [batch, H, W] -> [batch, HW]
        x: [batch, HW, HW]
        """
        x = torch.exp(x)

        weight = 1-mask
        weight = weight.view(weight.size(0), weight.size(1)*weight.size(2), -1)
        x = x*weight

        sum_x = x.sum(1)
        sum_x = sum_x.view(sum_x.size(0), -1, sum_x.size(1))
        x = x/sum_x
        return x

    def forward(self, x, mask):
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        mu_x = self.mu(x)

        theta_x = theta_x.permute(0, 2, 1)
        x = torch.bmm(theta_x, phi_x)
        x = self.mask_normalization(x, mask)
        x = torch.bmm(mu_x, x)

        x = self.nu(x)
        x = self.after_nonlocal(x)
        return x


class Mapping(nn.Module):
    def __init__(self, z_dim):
        super(Mapping, self).__init__()
        self.fc1 = nn.Linear(z_dim, 64*64*64)
        self.fc2 = nn.Linear(64*64*64, z_dim)

        self.preNonLocal = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU()
        )

        self.nonLocal = PartialNonLocal()

        self.postNonLocal = nn.Sequential(
            ResBlock(512), ResBlock(512), ResBlock(512), ResBlock(512), ResBlock(512), ResBlock(512),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, z, mask):
        """
        nonLocal_x : [batch, C, H, W]
        identical_x : [batch, C, H, W]
        mask : [batch, H, W]
        """
        h = self.fc1(z)
        h = h.view(h.size(0), 64, 64, 64)
        identical_x = self.preNonLocal(h)

        nonLocal_x = self.nonLocal(identical_x, mask)

        # x = (1-mask)*identical_x + mask*nonLocal_x
        x = identical_x*((1-mask).view(mask.size(0), -1, mask.size(1), mask.size(2))) + \
            nonLocal_x*(mask.view(mask.size(0), -1, mask.size(1), mask.size(2)))

        y = self.postNonLocal(x)
        y = y.view(y.size(0), -1)
        y = self.fc2(y)
        return y


class DiscriminatorLatent(nn.Module):
    def __init__(self, z_dim):
        super(DiscriminatorLatent, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


class DiscriminatorImage(nn.Module):
    def __init__(self):
        super(DiscriminatorImage, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # 256 * 256
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),  # 128 * 128
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 160, 4, 2, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(160, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8
        )

        self.output = nn.Sequential(
            nn.Conv2d(192, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        return self.output(l5), [l1, l2, l3, l4, l5]


class VGGBackbone(nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        return [l1, l2, l3, l4, l5]
