from PIL import Image
import mindspore
from mindspore.dataset.core import config
from mindspore.dataset.vision.py_transforms import HWC2CHW
from mindspore.train import model
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize
from scipy.io import loadmat
from os import name, path

import mindspore.dataset as ds
from mindspore import ParameterTuple
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset import vision
from mindspore.dataset import transforms

from mindspore import dtype as mstype

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations.array_ops import Padding
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import ipdb

def read_pacs_data(dataset_path, domain_name_list, split="train"):
    data_paths = []
    data_labels = []
    
    for domain_name in domain_name_list:
        split_file = path.join(dataset_path, "{}_{}_kfold.txt".format(domain_name, split))
        img_path = path.join('dataset', 'PACS')
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(img_path, data_path)
                label = int(label)-1
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels

    
class DatasetGenerator_test:
    def __init__(self, data_paths, data_labels, transforms, target_transforms):
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        label = self.data_labels[index]
        img1 = self.transforms(img)
        label = self.target_transforms(label)

        return img1, label

    def __len__(self):
        return len(self.data_paths)
    
def pacs_dataset_read(base_path, source_domain_list, target_domain_list, batch_size, target_flg=False):
    print('load source domain dataset: {}'.format(source_domain_list))
    print('load target domain dataset: {}'.format(target_domain_list))  
    
    dataset_path = path.join(base_path, 'data', 'correct_txt_lists')
#     ipdb.set_trace()
    train_split = 'train'
    test_split = 'test'
    train_data_paths, train_data_labels = read_pacs_data(dataset_path, source_domain_list, split=train_split)
    test_data_paths, test_data_labels = read_pacs_data(dataset_path, target_domain_list, split=test_split)

    transforms_test = transforms.Compose(
            [vision.Resize([224, 224], interpolation=Inter.LINEAR),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
            ])
    
    label_transform = transforms.TypeCast(mindspore.int32)

    train_dataset = DatasetGenerator_test(train_data_paths, train_data_labels,transforms_test, label_transform)
    test_dataset = DatasetGenerator_test(test_data_paths, test_data_labels, transforms_test, label_transform)
    
    dataset_train = ds.GeneratorDataset(train_dataset, ['data', 'label'], shuffle=True)
    dataset_test = ds.GeneratorDataset(test_dataset, ['data', 'label'], shuffle=False)
    
    dataset_train = dataset_train.batch(batch_size=batch_size, drop_remainder=True)
    dataset_test = dataset_test.batch(batch_size=batch_size, drop_remainder=True)

    return dataset_train, dataset_test
