import numpy as np
import torch
from torch.utils.data import Dataset
from .Utils import *

class Train_Dataset(Dataset):
    def __init__(self, file_address, training_paths, roi, im_size, stage):
        self.file_address = file_address
        self.training_paths = training_paths
        self.roi = roi
        self.im_size = im_size
        self.stage = stage
    def __len__(self):
        return len(self.training_paths)
    def __getitem__(self, idx):
        image, label = read_training_inputs(self.training_paths[idx], self.file_address, self.roi, self.im_size, self.stage)
        label = np.array([label]).astype('float32')
        image = np.array([image]).astype('float32')
        return [torch.from_numpy(image), torch.from_numpy(label), self.training_paths[idx]]

class Test_Dataset(Dataset):
    def __init__(self, testing_paths, image_path, output_path, roi, im_size,  stage):
        self.testing_paths = testing_paths
        self.image_path = image_path
        self.output_path = output_path
        self.roi = roi
        self.im_size = im_size
        self.stage = stage
    def __len__(self):
        return len(self.testing_paths)
    def __getitem__(self, idx):
        image, read_info = read_testing_inputs(self.testing_paths[idx], self.image_path, self.output_path, self.roi, self.im_size, self.stage)
        image = np.array([image]).astype('float32')
        return [torch.from_numpy(image), read_info, self.testing_paths[idx]]