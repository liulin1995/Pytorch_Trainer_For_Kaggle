import os
import torch as t
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, root, train_mode=True, transform=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transform
        self.train_mode = train_mode

    def __getitem__(self, item):
        img_path = self.imgs[item]

        if self.train_mode:
            # label = t.FloatTensor([0]*2019)
            class_index = int(img_path.split('_')[-1].split('.')[0])
            # label[class_index] = 1.0

            PIL_image = Image.open(img_path)
            PIL_image = PIL_image.convert('RGB')
            if self.transform:
                PIL_image = self.transform(PIL_image)

            return PIL_image, t.FloatTensor([class_index]).long()
        else:
            file_name = img_path.split('\\')[-1]
            PIL_image = Image.open(img_path)
            PIL_image = PIL_image.convert('RGB')
            if self.transform:
                PIL_image = self.transform(PIL_image)

            return PIL_image, file_name

    def __len__(self):
        return len(self.imgs)
