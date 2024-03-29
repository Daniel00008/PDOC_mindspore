from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# from data import StandardDataset
from data.JigsawLoader import _dataset_info
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset
import ipdb

# mnist = 'mnist'
# mnist_m = 'mnist_m'
# svhn = 'svhn'
# synth = 'synth'
# usps = 'usps'

# vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
# pacs_datasets = ["test_art_painting_stage1.txt", "cartoon", "photo", "sketch"]
# office_datasets = ["amazon", "dslr", "webcam"]
# digits_datasets = [mnist, mnist, svhn, usps]
# available_datasets =  pacs_datasets
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

# dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
#                mnist_m: (0.2384788, 0.22375608, 0.24496263),
#                svhn: (0.1951134, 0.19804622, 0.19481073),
#                synth: (0.29410212, 0.2939651, 0.29404707),
#                usps: (0.25887518, 0.25887518, 0.25887518),
#                }

# dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
#                 mnist_m: (0.45920207, 0.46326601, 0.41085603),
#                 svhn: (0.43744073, 0.4437959, 0.4733686),
#                 synth: (0.46332872, 0.46316052, 0.46327512),
#                 usps: (0.17025368, 0.17025368, 0.17025368),
#                 }


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    
    ipdb.set_trace()
    if args.task == 'PACS':
        for dname in dataset_list:
            # name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
            name_train, labels_train = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_train_kfold.txt' % dname))
#             import ipdb
#             ipdb.set_trace()
            name_val, labels_val = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_crossval_kfold.txt' % dname))

            train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                          tile_transformer=tile_transformer, jig_classes=30, bias_whole_image=args.bias_whole_image)
            datasets.append(train_dataset)
            val_datasets.append(
                JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
                                  patches=patches, jig_classes=30))
#     elif args.task == 'VLCS':
#         for dname in dataset_list:
# #             name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_train.txt' % dname), args.val_size)
#             name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_train.txt' % dname), 0)
#             train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches, img_transformer=img_transformer,
#                                              tile_transformer=tile_transformer, jig_classes=30,
#                                              bias_whole_image=args.bias_whole_image)
#             datasets.append(train_dataset)
#             val_datasets.append(
#                 JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
#                                      patches=patches, jig_classes=30))
#     elif args.task == 'HOME':
#         for dname in dataset_list:
#             name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_full.txt' % dname), 0)#args.val_size
#             train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches,
#                                              img_transformer=img_transformer,
#                                              tile_transformer=tile_transformer, jig_classes=30,
#                                              bias_whole_image=args.bias_whole_image)
#             datasets.append(train_dataset)
#             val_datasets.append(
#                 JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
#                                      patches=patches, jig_classes=30))
            
#     elif args.task == 'DNET':
#         for dname in dataset_list:
#             name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'correct_txt_lists','domainnet', '%s_train.txt' % dname), 0)#args.val_size
#             train_dataset = JigsawNewDataset(args, name_train, labels_train, patches=patches,
#                                              img_transformer=img_transformer,
#                                              tile_transformer=tile_transformer, jig_classes=30,
#                                              bias_whole_image=args.bias_whole_image)
#             datasets.append(train_dataset)
#             val_datasets.append(
#                 JigsawTestNewDataset(args, name_val, labels_val, img_transformer=get_val_transformer(args),
#                                      patches=patches, jig_classes=30))
            
    else:
        raise NotImplementedError('DATA LOADER NOT IMPLEMENTED.')

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(args, patches=False):

    dataset_list = args.target
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_tr = get_val_transformer(args)
    for dname in dataset_list:
        if args.task == 'PACS':
            names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test_kfold.txt' % dname))
#         elif args.task == 'VLCS':
#             names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test.txt' % dname))
#         elif args.task == 'HOME':
#             names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_full.txt' % dname))
#         elif args.task == 'DNET':
#             names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', 'domainnet', '%s_test.txt' % dname))
        else:
            raise NotImplementedError('TEST DATA LOADER NOT IMPLEMENTED.')
        
        val_dataset = JigsawTestNewDataset(args,names, labels, patches=patches, img_transformer=img_tr, jig_classes=30)  
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_multiple_val_dataloader(args, patches=False):
    loaders = []

    for dname in args.target:
        if args.task == 'PACS':
            names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test_kfold.txt' % dname))
#         elif args.task == 'VLCS':
#             names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_test.txt' % dname))
#         elif args.task == 'HOME':
#             names, labels = _dataset_info(join(dirname(__file__), 'correct_txt_lists', '%s_full.txt' % dname))
        else:
            raise NotImplementedError('TEST DATA LOADER NOT IMPLEMENTED.')
        img_tr = get_val_transformer(args)
        val_dataset = JigsawTestNewDataset(args,names, labels, patches=patches, img_transformer=img_tr, jig_classes=30)
        if args.limit_target and len(val_dataset) > args.limit_target:
            val_dataset = Subset(val_dataset, args.limit_target)
            print("Using %d subset of val dataset" % args.limit_target)
        dataset = ConcatDataset([val_dataset])
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        loaders.append(loader)
    return loaders


# JIGSAW
def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale),interpolation=transforms.InterpolationMode.NEAREST)]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tile_tr = []
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)



def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

