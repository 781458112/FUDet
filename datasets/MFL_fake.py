import os
from enum import Enum
import numpy as np
import PIL
import torch
from torchvision import transforms
import scipy.io as scio
from abnormal import abnormal
import re
import matplotlib.pyplot as plt
_CLASSNAMES = [
    "MFL",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MFLfakeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.abnormal=abnormal()
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_fake_img = [
            transforms.Resize(resize),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            # transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_fake_img = transforms.Compose(self.transform_fake_img)
        self.transform_mask = [
            # transforms.Resize(resize),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def colormap(self,data, map_scale):
        data = np.array(data)
        averdata = np.mean(data)
        stddata = np.std(data)
        maxvalue = averdata + 5 * stddata
        minvalue = averdata - 2 * stddata
        data = (data - minvalue) / (maxvalue - minvalue)
        data = np.ceil(data * 1023)
        data[data > 1023] = 1023
        data[data < 0] = 0
        finaldata = np.transpose(data)
        finaldata = finaldata.astype(np.int16)
        out = map_scale[finaldata, :]
        return out

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path,mat_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        if mat_path is not None:
            mat = scio.loadmat(mat_path)
            KeyList = list(mat.keys())
            mat_image = np.array(mat[KeyList[3]][:, 0:])
            map = scio.loadmat('./jet.mat')
            map = map['map_scale']
            fake_image = self.abnormal.out(mat_image)
            out = self.colormap(data=fake_image, map_scale=map)
            image_old = torch.tensor(out).float().permute(2, 0, 1).contiguous()
            ffake_image = self.transform_fake_img(image_old)
        else:
            ffake_image = torch.zeros([3,180,180])

        # img2 = [(x - np.min(image)) / (np.max(image) - np.min(image)) * 255 for x in mat_image]
        # img3 = np.transpose(img2)



        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "fake_image":ffake_image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        mats_per_class={}
        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            matpath = os.path.join(self.source, classname, "mat")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            mats_per_class[classname] = {}
            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path),key=natural_sort_key)
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path), key=natural_sort_key)
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None
                anomaly_mat_path = matpath
                anomaly_mat_files = sorted(os.listdir(anomaly_mat_path),key=natural_sort_key)
                mats_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mat_path, x) for x in anomaly_mat_files
                    ]
        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys(),key=natural_sort_key):
            for anomaly in sorted(imgpaths_per_class[classname].keys(),key=natural_sort_key):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        data_tuple.append(None)
                    else:
                        data_tuple.append(None)
                        data_tuple.append(mats_per_class[classname][anomaly][i])
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate