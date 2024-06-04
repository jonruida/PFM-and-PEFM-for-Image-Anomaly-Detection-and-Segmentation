import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

# Custom class names based on your dataset structure
CUSTOM_CLASS_NAMES = ['good']

class CustomDataset(Dataset):
    def __init__(self, root_path='content/data/', class_name='good', is_train=True, resize=256, trans=None, LOAD_CPU=False):
        assert class_name in CUSTOM_CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CUSTOM_CLASS_NAMES)

        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize

        # Load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # Set transforms
        if trans is None:
            self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                          T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])
        else:
            self.transform_x = trans

        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.ToTensor()])

        self.load_cpu = LOAD_CPU
        self.len = len(self.x)
        self.x_cpu = []
        self.y_cpu = []
        self.name = []
        self.mask_cpu = []
        if self.load_cpu:
            for i in range(self.len):
                names = self.x[i].split("/")
                name = names[-2] + "_" + names[-1]
                self.name.append(name)
                x = Image.open(self.x[i]).convert('RGB')
                x = self.transform_x(x)
                self.x_cpu.append(x)

                if self.y[i] == 0:
                    mask = torch.zeros([1, self.resize, self.resize])
                else:
                    mask = Image.open(self.mask[i])
                    mask = self.transform_mask(mask)
                self.mask_cpu.append(mask)
                self.y_cpu.append(self.y[i])

    def __getitem__(self, idx):
        if self.load_cpu:
            x, y, mask, name = self.x_cpu[idx], self.y_cpu[idx], self.mask_cpu[idx], self.name[idx]
        else:
            x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
            names = x.split("\\")
            name = names[-2] + "_" + names[-1]
            x = Image.open(x).convert('RGB')
            x = self.transform_x(x)

            if y == 0:
                mask = torch.zeros([1, self.resize, self.resize])
            else:
                mask = Image.open(mask)
                mask = self.transform_mask(mask)

        return x, y, mask, name

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.root_path, phase, self.class_name)

        # Load images
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                 for f in os.listdir(img_dir)
                                 if f.endswith('.png')])
        x.extend(img_fpath_list)

        # Since your dataset only contains 'good' class, all labels are 0
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be the same'

        return list(x), list(y), list(mask)

if __name__ == "__main__":
    dataset_ = CustomDataset()
    a = dataset_[0]
    print(a)
