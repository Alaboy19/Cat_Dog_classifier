from torch.utils.data import Dataset
import cv2
from config import *
import torch
from torchvision.transforms import functional as f


class Cat_Dog_Dataset(Dataset):
    def __init__(self, dataframe, augmentations):
        self.dataframe = dataframe
        self.augmentations = augmentations

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        filename, label = self.dataframe[index]


        try:
            img = cv2.imread(filename)
        except Exception as e:
            print(e.what(), filename)

        img = cv2.resize(img, (image_w, image_h))

        augmented_image = self.augmentations(image=img)['image']

        return f.to_tensor(augmented_image), label
