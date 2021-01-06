import numpy as np

import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import Dataset

class MyDataset(Dataset) :
    def __init__(self, x_data, y_data, transform=None) :
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index) :
        sample = self.x_data[index], self.y_data[index]
        if self.transform :
            sample = self.transform(sample)
        return sample

    def __len__(self) :
        return self.len

class MyTransform_numpy :
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2, 0, 1) #channel, height, width
        lables = torch.FloatTensor(labels)

        return inputs, labels

class MyTransform_PIL :
    def __call__(self, sample) :
        inputs, labels = sample
        transf = tr.Compose[
            tr.ToPILImage()
            , tr.Resize(128)
            , tr.ToTensor()
            , tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        outputs = trnasf(inputs)
        return outputs, lables