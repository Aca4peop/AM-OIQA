import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class Dataset_Enhanced(Dataset):
   
    def __init__(self, csv_path, database, train):
        """
        Dataset class for loading and processing the data with data argument.
        Args:
            csv_path: path to the csv file containing the index of images.
            database: name of the database.
            train: boolean indicating whether the dataset is for training or testing.
        """
        column_name = ['img', 'mos']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_name, index_col=False, encoding='utf-8-sig')
        self.X = tmp_df['img']
        if database == 'CVIQ':
            self.X = tmp_df['img'].apply(lambda x:f'{x:03d}')
        self.Y = tmp_df['mos']
        self.length = len(tmp_df)
        self.database = database
        self.train = train


        transformer = []
        if self.train == True:
            transformer.extend([
                    transforms.RandomCrop(256),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5)
                    ])
        else:
            transformer.append(transforms.CenterCrop(256))
        transformer.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)])
        self.transform = transforms.Compose(transformer)
    def __getitem__(self, index):

        ori_img_path = self.X.iloc[index]
        root = ori_img_path

        vp_list = []
        img_path = f'/home1/lyq/Dataset/{self.database}/resized_cubic_equator/{root}'
        direction_list = ['A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png']
        for item in direction_list:
            with open( img_path + '_'+item,'rb') as f:
                img = Image.open(f).convert('RGB')
                img_tf = self.transform(img)
                vp_list.append(img_tf)
        y_mos = self.Y.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))
        return vp_list, y_label

    def __len__(self):
        return self.length


class Dataset(Dataset):
    def __init__(self, csv_path, database, train):
        """
            Dataset class for loading and processing the data without data argument.
            Args:
                csv_path: path to the csv file containing the index of images.
                database: name of the database.
                train: boolean indicating whether the dataset is for training or testing.
        """
        column_name =   ['img', 'mos']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_name, index_col=False, encoding='utf-8-sig')
        self.X = tmp_df['img']
        if database == 'CVIQ':
            self.X = tmp_df['img'].apply(lambda x:f'{x:03d}')
        self.Y = tmp_df['mos']
        self.length = len(tmp_df)
        self.database = database
        self.train = train
        transformer = []
        transformer.extend([
                transforms.Resize(256),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)])
        self.transform = transforms.Compose(transformer)

    def __getitem__(self, index):

        ori_img_path = self.X.iloc[index]
        root = ori_img_path

        vp_list = []
        img_path = f'/home1/lyq/Dataset/{self.database}/resized_cubic_equator/{root}'
        direction_list = ['A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png']
        for item in direction_list:
            item_img_path = img_path + '_'+item
            img = Image.open(item_img_path)
            img = img.convert('RGB')
            img = self.transform(img)
            vp_list.append(img)
        y_mos = self.Y.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        return vp_list, y_label

    def __len__(self):
        return self.length
    


def get_dataset(csv_path, database, train):
    if database in ['CVIQ', 'OIQ10K']:
        dataset = Dataset(csv_path, database, train)
    else:
        dataset = Dataset_Enhanced(csv_path, database, train)
    return dataset
