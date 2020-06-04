import os
import torch
import torchvision
# import model
import complex_model as model
import loss
from torch.utils.data import DataLoader
from dataset import YDS, RXDS, XYDS
import argparse
import itertools

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--device", default="cpu", type=str, help="the computation device (cpu or cuda)")
parser.add_argument("-y", "--y_root_dir", default="./data/yimg", type=str)
parser.add_argument("-r", "--r_root_dir", default='./data/rimg', type=str)
parser.add_argument("-x", "--x_root_dir", default='./data/ximg', type=str)
parser.add_argument("-s", "--save_dir", default='./checkpoint', type=str)
parser.add_argument("-b", "--batch_size", default=8, type=int)
parser.add_argument("-z", "--z_dim", default=30, type=int)

parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("-v", "--lr_vae", default=1e-6, type=float)
parser.add_argument("-e", "--lr_encoder", default=1e-6, type=float)
parser.add_argument("-g", "--lr_generator", default=1e-6, type=float)
parser.add_argument("-i", "--lr_discimg", default=1e-5, type=float)
parser.add_argument("-l", "--lr_disclatent", default=3e-6, type=float)
parser.add_argument("-m", "--lr_mapping", default=1e-7, type=float)

parser.add_argument("--epoch_y", default=50, type=int)
parser.add_argument("--epoch_rx", default=50, type=int)
parser.add_argument("--epoch_xy", default=50, type=int)
parser.add_argument("--save_epoch", default=10, type=int)

# parser.add_argument("--alpha1", default=2, type=float)
# parser.add_argument("--alpha2", default=10, type=float)
# parser.add_argument("--lambda1", default=60, type=float)
# parser.add_argument("--lambda2", default=10, type=float)
parser.add_argument("--kl_weight", default=10, type=float)
parser.add_argument("--rec_weight", default=10, type=float)
parser.add_argument("--gen_weight", default=1, type=float)
parser.add_argument("--disc_weight", default=1, type=float)
args = parser.parse_args()


def train_y(device, transforms):
    y_dataset = YDS(args.y_root_dir, transform=transforms)
    y_dataloader = DataLoader(y_dataset, batch_size=args.batch_size, shuffle=True)

    dict_path = os.path.join(args.save_dir, "y_training.pth")
    load_dict = torch.load(dict_path) if os.path.exists(dict_path) else None

    # Ey = model.Encoder(args.z_dim)
    # Gy = model.Generator(args.z_dim)
    Ey = model.Encoder()
    Gy = model.Generator()
    Dy = model.DiscriminatorImage()

    if load_dict:
        print("load the trained model")
        Ey.load_state_dict(load_dict['ey'])
        Gy.load_state_dict(load_dict['gy'])
        Dy.load_state_dict(load_dict['dy'])

    Ey.to(device)
    Gy.to(device)
    Dy.to(device)

    optimizer_vae = torch.optim.Adam(
        itertools.chain(
            filter(lambda p: p.requires_grad, Ey.parameters()),
            filter(lambda p: p.requires_grad, Gy.parameters())
        ),
        lr=args.lr,
        betas=(0.0, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Dy.parameters()),
        lr=args.lr,
        betas=(0.0, 0.999)
    )
    # optimizer_ey = torch.optim.Adam(Ey.parameters(), lr=args.lr_encoder)
    # optimizer_gy = torch.optim.Adam(Gy.parameters(), lr=args.lr_generator)
    # optimizer_dy = torch.optim.Adam(Dy.parameters(), lr=args.lr_discimg)

    for epoch in range(1, args.epoch_y+1):
        vae_loss_sum = 0.
        vae_kl_loss = 0.
        vae_rec_loss = 0.
        vae_gan_lsloss = 0.

        disc_loss_sum = 0.
        for idx, y in enumerate(y_dataloader):
            y = y.to(device)
            # train VAE
            optimizer_vae.zero_grad()

            # zy = Ey(y)
            # y_ = Gy(zy)
            # sy_, _ = Dy(y_)
            zy, _ = Ey(y)
            y_ = Gy(zy)
            sy_, _ = Dy(y_)

            # KL_loss = loss.latent_loss(Ey.mu, Ey.logvar)
            KL_loss = Ey.kl_divergence.mean()
            reconstruct_loss = loss.VAE_reconstruct_loss(y_, y)
            lsloss_gen = loss.LSLoss_gen(sy_)

            # vae_loss = args.alpha1 * KL_loss + args.alpha2 * reconstruct_loss
            vae_loss = args.kl_weight * KL_loss + args.rec_weight * reconstruct_loss + args.gen_weight * lsloss_gen
            # vae_loss = KL_loss + reconstruct_loss + lsloss_gen
            vae_loss.backward()

            optimizer_vae.step()
            vae_loss_sum += vae_loss.item()
            vae_kl_loss += KL_loss.item()
            vae_rec_loss += reconstruct_loss.item()
            vae_gan_lsloss += lsloss_gen.item()

            # train discriminator
            optimizer_d.zero_grad()
            zy, _ = Ey(y)
            y_ = Gy(zy)
            sy_, _ = Dy(y_)

            lsloss_disc_fake = loss.LSLoss_disc_fake(sy_)
            disc_loss_sum += lsloss_disc_fake.item()

            sy, _ = Dy(y)
            lsloss_disc_real = loss.LSLoss_disc_real(sy)
            disc_loss_sum += lsloss_disc_real.item()

            lsloss_disc = lsloss_disc_fake + lsloss_disc_real
            lsloss_disc = lsloss_disc * args.disc_weight
            lsloss_disc.backward()
            optimizer_d.step()

            if (idx+1) % 50 == 0:
                # print("--idx %d: VAE LOSS: %f" % (idx+1, vae_loss.item()))
                print(
                    "--idx %d: VAE LOSS: %f, KL LOSS: %f, REC LOSS: %f, GEN LOSS: %f, DISC LOSS: %f " %
                    (idx+1, vae_loss.item(), KL_loss.item(), reconstruct_loss.item(), lsloss_gen.item(), lsloss_disc.item())
                    )
                # print("--idx %d: VAE LOSS: %f, DISC LOSS: %f " % (idx+1, vae_loss.item(), lsloss_disc.item()))
        print(
            "epoch %d: VAE LOSS: %f, KL LOSS: %f, REC LOSS: %f, GEN LOSS: %f, DISC LOSS: %f " %
            (epoch, vae_loss_sum, vae_kl_loss, vae_rec_loss, vae_gan_lsloss, disc_loss_sum)
            )
        if epoch % args.save_epoch == 0:
            save_dict = {
                'ey': Ey.state_dict(),
                'gy': Gy.state_dict(),
                'dy': Dy.state_dict(),
                }
            torch.save(save_dict, os.path.join(args.save_dir, "y_training.pth"))
            print("This checkpoint for y training has been saved.")

    print("Training process for y has finished.")
    save_dict = {
        'ey': Ey.state_dict(),
        'gy': Gy.state_dict(),
        'dy': Dy.state_dict(),
        }
    torch.save(save_dict, os.path.join(args.save_dir, "y_training.pth"))
    print("This checkpoint for y training has been saved.")


def train_rx(device, transforms):
    rx_dataset = RXDS(args.r_root_dir, args.x_root_dir, transform=transforms)
    rx_dataloader = DataLoader(rx_dataset, batch_size=args.batch_size, shuffle=True)

    dict_path = os.path.join(args.save_dir, "rx_training.pth")
    load_dict = torch.load(dict_path) if os.path.exists(dict_path) else None

    Erx, Grx = model.Encoder(args.z_dim), model.Generator(args.z_dim)
    Drx, Dz = model.DiscriminatorImage(), model.DiscriminatorLatent(args.z_dim)

    if load_dict:
        Erx.load_state_dict(load_dict['erx'])
        Grx.load_state_dict(load_dict['grx'])
        Drx.load_state_dict(load_dict['drx'])
        Dz.load_state_dict(load_dict['dz'])

    Erx.to(device)
    Grx.to(device)
    Drx.to(device)
    Dz.to(device)

    optimizer_erx = torch.optim.Adam(Erx.parameters(), lr=args.lr_encoder)
    optimizer_grx = torch.optim.Adam(Grx.parameters(), lr=args.lr_generator)
    optimizer_drx = torch.optim.Adam(Drx.parameters(), lr=args.lr_discimg)
    optimizer_dz = torch.optim.Adam(Dz.parameters(), lr=args.lr_disclatent)

    for epoch in range(1, args.epoch_rx+1):
        vae_loss_sum = 0.
        drx_loss_sum = 0.
        dz_loss_sum = 0.
        for idx, (r, x) in enumerate(rx_dataloader):
            isx = True
            vae_loss_temp = 0.
            drx_loss_temp = 0.
            dz_loss_temp = 0.
            for img in [x, r]:
                # train VAE
                optimizer_erx.zero_grad()
                optimizer_grx.zero_grad()
                z_img = Erx(img)
                img_ = Grx(z_img)
                s_img_, _ = Drx(img_)
                KL_loss = loss.latent_loss(Erx.mu, Erx.logvar)
                reconstruct_loss = loss.VAE_reconstruct_loss(img_, img)
                lsloss_gen = loss.LSLoss_gen(s_img_)
                vae_loss = KL_loss + args.alpha * reconstruct_loss + lsloss_gen
                if isx:
                    isx = False
                    szx = Dz(z_img)
                    ganloss_gen = loss.GAN_latent_loss_gen(szx)
                    vae_loss = vae_loss + ganloss_gen
                vae_loss.backward()

                optimizer_erx.step()
                optimizer_grx.step()
                vae_loss_temp += vae_loss.item()

                # train discriminator Drx
                optimizer_drx.zero_grad()
                z_img = Erx(img)
                img_ = Grx(z_img)
                s_img_, _ = Drx(img_)

                lsloss_disc_fake = loss.LSLoss_disc_fake(s_img_)

                s_img, _ = Drx(img)
                lsloss_disc_real = loss.LSLoss_disc_real(s_img)

                lsloss_disc = lsloss_disc_fake + lsloss_disc_real
                lsloss_disc.backward()
                optimizer_drx.step()

                drx_loss_temp = drx_loss_temp + lsloss_disc_fake.item() + lsloss_disc_real.item()

            # train discriminator Dz
            optimizer_dz.zero_grad()
            # using x
            zx = Erx(x)
            szx = Dz(zx)
            ganloss_disc_fake = loss.GAN_latent_loss_disc_fake(szx)

            # using r
            zr = Erx(r)
            szr = Dz(zr)
            ganloss_disc_real = loss.GAN_latent_loss_disc_real(szr)

            ganloss_disc = ganloss_disc_fake + ganloss_disc_real
            ganloss_disc.backward()
            optimizer_dz.step()

            dz_loss_temp = dz_loss_temp + ganloss_disc.item()

            vae_loss_sum += vae_loss_temp
            drx_loss_sum += drx_loss_temp
            dz_loss_sum += dz_loss_temp

            print("idx %d: VAE LOSS: %f, DRX LOSS: %f, DZ LOSS: %f " % (idx, vae_loss_temp, drx_loss_temp, dz_loss_temp))

        print("epoch %d: VAE LOSS: %f, DRX LOSS: %f, DZ LOSS: %f " % (epoch, vae_loss_sum, drx_loss_sum, dz_loss_sum))

    print("Training process for r&x has finished.")
    save_dict = {
        'erx': Erx.state_dict(),  # 'erx_optimizer': optimizer_erx.state_dict(),
        'grx': Grx.state_dict(),  # 'grx_optimizer': optimizer_grx.state_dict(),
        'drx': Drx.state_dict(),  # 'drx_optimizer': optimizer_drx.state_dict(),
        'dz': Dz.state_dict(),  # 'dz_optimizer': optimizer_dz.state_dict(),
        }
    torch.save(save_dict, os.path.join(args.save_dir, "rx_training.pth"))
    print("This checkpoint for r&x training has been saved.")


def train_xy(device, transforms):
    xy_dataset = XYDS(args.x_root_dir, args.y_root_dir, transforms)
    xy_dataloader = DataLoader(xy_dataset, batch_size=args.batch_size, shuffle=True)

    if not os.path.exists(os.path.join(args.save_dir, 'y_training.pth')):
        print("The training of y is not complete.")
        return
    y_load_dict = torch.load(os.path.join(args.save_dir, 'y_training.pth'))

    if not os.path.exists(os.path.join(args.save_dir, 'rx_training.pth')):
        print("The training of r&x is not complete.")
        return
    rx_load_dict = torch.load(os.path.join(args.save_dir, 'rx_training.pth'))

    dict_path = os.path.join(args.save_dir, "xy_training.pth")
    load_dict = torch.load(dict_path) if os.path.exists(dict_path) else None

    # there should be a state load for segmentation network

    Ey = model.Encoder(args.z_dim)
    Gy = model.Generator(args.z_dim)
    Erx = model.Encoder(args.z_dim)

    Ey.load_state_dict(y_load_dict['ey'])
    Gy.load_state_dict(y_load_dict['gy'])
    Erx.load_state_dict(rx_load_dict['erx'])

    M = model.Mapping(args.z_dim)
    DM = model.DiscriminatorImage()

    if load_dict:
        M.load_state_dict(load_dict['m'])
        DM.load_state_dict(load_dict['dm'])

    VGG = model.VGGBackbone()
    vggbb_state = torch.load(os.path.join(args.save_dir, 'vggbb.pth'))
    VGG.load_state_dict(vggbb_state)

    Ey.to(device)
    Gy.to(device)
    Erx.to(device)
    M.to(device)
    DM.to(device)
    VGG.to(device)

    optimizer_m = torch.optim.Adam(M.parameters(), lr=args.lr_mapping)
    optimizer_dm = torch.optim.Adam(DM.parameters(), lr=args.lr_discimg)

    for epoch in range(1, args.epoch_xy+1):
        m_loss_sum = 0.
        dm_loss_sum = 0.
        for idx, (x, y) in enumerate(xy_dataloader):
            m_loss_temp = 0.
            dm_loss_temp = 0.
            # train M
            optimizer_m.zero_grad()
            zx = Erx(x)
            zy = Ey(y)
            mask = torch.zeros(args.batch_size, 64, 64)  # this mask should be gotten from a segmentation network
            zx_ = M(zx, mask)
            x_ = Gy(zx_)
            y_ = Gy(zy)

            latent_space_loss = loss.latent_space_loss(zx_, zy)
            sx_, fm_x_d = DM(x_)
            sy_, fm_y_d = DM(y_)
            fm_x_vgg = VGG(x_)
            fm_y_vgg = VGG(y_)
            lsloss_gen = loss.LSLoss_gen(sx_)
            fm_loss = loss.feature_matching_loss(fm_x_d, fm_y_d, fm_x_vgg, fm_y_vgg)

            mapping_loss = args.lambda1 * latent_space_loss + lsloss_gen + args.lambda2 * fm_loss
            mapping_loss.backward()
            optimizer_m.step()

            m_loss_temp = mapping_loss.item()

            # train DM
            optimizer_dm.zero_grad()
            zx = Erx(x)
            zy = Ey(y)
            mask = torch.zeros(args.batch_size, 64, 64)  # this mask should be gotten from a segmentation network
            zx_ = M(zx, mask)
            x_ = Gy(zx_)
            y_ = Gy(zy)
            sx_, _ = DM(x_)
            sy_, _ = DM(y_)

            lsloss_disc_fake = loss.LSLoss_disc_fake(sx_)
            lsloss_disc_real = loss.LSLoss_disc_real(sy_)

            lsloss_disc = lsloss_disc_fake + lsloss_disc_real
            lsloss_disc.backward()
            optimizer_dm.step()

            dm_loss_temp = lsloss_disc.item()

            m_loss_sum += m_loss_temp
            dm_loss_sum += dm_loss_temp
            print("idx %d: MAPPING LOSS: %f, DM LOSS: %f " % (idx, m_loss_temp, dm_loss_temp))

        print("epoch %d: MAPPING LOSS: %f, DM LOSS: %f " % (epoch, m_loss_sum, dm_loss_sum))

    print("Training process for x&y has finished.")
    save_dict = {
        'm': M.state_dict(),  # 'm_optimizer': optimizer_m.state_dict(),
        'dm': DM.state_dict(),  # 'dm_optimizer': optimizer_dm.state_dict(),
        }
    torch.save(save_dict, os.path.join(args.save_dir, "xy_training.pth"))
    print("This checkpoint for x&y training has been saved.")


def train():
    device = torch.device(args.device)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Training for y begin!")
    train_y(device, transforms)
    # print("Training for r&x begin!")
    # train_rx(device, transforms)
    # print("Training for x&y begin!")
    # train_xy(device, transforms)


if __name__ == "__main__":
    train()
