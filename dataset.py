# import torch
from PIL import Image, ImageFile
import imageio
import numpy as np
import torchvision
import os
# import cv2
from torch.utils.data import Dataset, DataLoader


class YDS(Dataset):
    def __init__(self, root_dir, transform=None):
        super(YDS, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_path = os.path.join(self.root_dir, img_path)
        # img = cv2.imread(img_path)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_pil = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img_pil)
        img_pil.close()

        return img

    def __len__(self):
        return len(self.images)


class RXDS(Dataset):
    def __init__(self, r_root_dir, x_root_dir, transform=None):
        super(RXDS, self).__init__()
        self.r_root_dir = r_root_dir
        self.x_root_dir = x_root_dir
        self.transform = transform
        self.r_images = os.listdir(self.r_root_dir)
        self.x_images = os.listdir(self.x_root_dir)

    def __getitem__(self, idx):
        r_img_path = self.r_images[idx]
        x_img_path = self.x_images[idx]
        r_img_path = os.path.join(self.r_root_dir, r_img_path)
        x_img_path = os.path.join(self.x_root_dir, x_img_path)
        # r_img = cv2.imread(r_img_path)
        # x_img = cv2.imread(x_img_path)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        r_img_pil = Image.open(r_img_path).convert('RGB')
        x_img_pil = Image.open(x_img_path).convert('RGB')
        if self.transform:
            x_img = self.transform(x_img_pil)
            r_img = self.transform(r_img_pil)
        x_img_pil.close()
        r_img_pil.close()

        return (r_img, x_img)

    def __len__(self):
        assert(len(self.r_images) == len(self.x_images))
        return len(self.r_images)


class XYDS(Dataset):
    def __init__(self, x_root_dir, y_root_dir, transform=None):
        super(XYDS, self).__init__()
        self.y_root_dir = y_root_dir
        self.x_root_dir = x_root_dir
        self.transform = transform
        self.y_images = os.listdir(self.y_root_dir)
        self.x_images = os.listdir(self.x_root_dir)

    def __getitem__(self, idx):
        y_img_path = self.y_images[idx]
        x_img_path = "%s_in.png" % y_img_path.strip().split('_')[0]
        y_img_path = os.path.join(self.y_root_dir, y_img_path)
        x_img_path = os.path.join(self.x_root_dir, x_img_path)
        # y_img = cv2.imread(y_img_path)
        # x_img = cv2.imread(x_img_path)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        y_img_pil = Image.open(y_img_path).convert('RGB')
        x_img_pil = Image.open(x_img_path).convert('RGB')
        if self.transform:
            x_img = self.transform(x_img_pil)
            y_img = self.transform(y_img_pil)
        x_img_pil.close()
        y_img_pil.close()

        return (x_img, y_img)

    def __len__(self):
        assert(len(self.y_images) == len(self.x_images))
        return len(self.y_images)


def save_img(tensor_data, save_path):
    numpy_img = tensor_data.cpu().float().numpy()
    numpy_img = (np.transpose(numpy_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    # numpy_img = np.transpose(numpy_img, (1, 2, 0)) * 255.0
    numpy_img = numpy_img.astype(np.uint8)
    imageio.imwrite(save_path, numpy_img)


if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    yds = YDS("./data/yimg", transform=transforms)
    data_loader = DataLoader(yds, batch_size=128, shuffle=True)
    for i, y_data in enumerate(data_loader):
        y_data = y_data*2-1
        for i in range(y_data.size(0)):
            img_name = "out_"+str(i)+".png"
            save_img(y_data[i].data, os.path.join("./data/out", img_name))

    # rxds = RXDS('./data/rimg', './data/ximg', transform=transforms)
    # data_loader = DataLoader(rxds, batch_size=128, shuffle=True)
    # for i, (r_data, x_data) in enumerate(data_loader):
    #     print(i, r_data.size(), x_data.size())

    # xyds = XYDS('./data/ximg', './data/yimg', transforms)
    # data_loader = DataLoader(xyds, batch_size=2, shuffle=True)
    # for i, (x_data, y_data) in enumerate(data_loader):
    #     print(i, x_data.size(), y_data.size())
