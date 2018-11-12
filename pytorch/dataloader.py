import pickle

import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms

from shared.rle import rle_decode

original_width = 768
original_height = 768


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def __call__(self, sample):
        image, loc_image = sample['image'], sample['loc_image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.dimensions:
            image, loc_image = self.resize_images(image, loc_image)
            self.binarize_mask(loc_image)

        image = self.transpose_and_scale_image(image)
        loc_image = self.reshape_and_scale_loc_image(loc_image)
        contains_ship = self.does_image_contain_ship(loc_image)
        return {'image': torch.from_numpy(image),
                'loc_image': torch.from_numpy(loc_image),
                'contains_ship': torch.from_numpy(contains_ship)}

    @staticmethod
    def does_image_contain_ship(loc_image):
        return np.asarray([1, 0], dtype=np.float32) if np.count_nonzero(loc_image) == 0 else np.asarray([0, 1],
                                                                                                        dtype=np.float32)

    @staticmethod
    def reshape_and_scale_loc_image(loc_image):
        return loc_image.reshape(1, loc_image.shape[0], loc_image.shape[1]).astype(np.float32) / 255

    @staticmethod
    def transpose_and_scale_image(image):
        return image.transpose((2, 0, 1)).astype(np.float32) / 255

    def resize_images(self, image, loc_image):
        height, width = self.dimensions
        image = transform.resize(image, (height, width), anti_aliasing=True, preserve_range=True)
        loc_image = transform.resize(loc_image, (height, width), anti_aliasing=False, preserve_range=True)
        return image, loc_image

    @staticmethod
    def binarize_mask(loc_image):
        loc_image[loc_image > 0] = 255


class ShipData(Dataset):
    def __init__(self, image_width, image_height, use_cache=True, dropna=True):
        self.image_width = image_width
        self.image_height = image_height
        self.use_cache = use_cache
        self.cache = {}
        self.dict_path = "shipdict.pickle"
        self.dataframe_path = "shipdatadf.pickle"

        self.load_ships_dataframe(dropna)
        self.load_merge_ships_dict()

        self.merge_ships_dataframe = self.ships_dataframe.drop('EncodedPixels', 1).drop_duplicates()

        self.transform = transforms.Compose([ToTensor((image_height, image_width))])

    def load_ships_dataframe(self, dropna):
        try:
            self.ships_dataframe = pickle.load(open(self.dataframe_path, 'rb'))
            print("loaded shipdatadf.pickle, delete file to regenerate")
        except:
            print("shipdatadf.pickle not yet generated creating now")
            self.ships_dataframe = pd.read_csv('data/train_ship_segmentations_v2.csv')
            if dropna:
                print('Dropping NANs')
                self.ships_dataframe = self.ships_dataframe.dropna()
            print("saving to shipdatadf.pickle")
            pickle.dump(self.ships_dataframe, open(self.dataframe_path, "wb"))
            print("saved")

    def load_merge_ships_dict(self):
        try:
            self.merge_ships_dict = pickle.load(open(self.dict_path, 'rb'))
            print("loaded shipdict.pickle, delete file to regenerate")
        except:
            print("shipdict.pickle not yet generated creating now")
            self.merge_ships_dict = self._merge_ships(self.ships_dataframe)
            print("saving to shipdict.pickle")
            pickle.dump(self.merge_ships_dict, open(self.dict_path, "wb"))
            print("saved")

    def __len__(self):
        return len(self.merge_ships_dict) - 1

    def _get_blank_sample(self):
        image = np.zeros((self.image_height, self.image_width, 3))
        loc_image = rle_decode([], original_width, original_height)
        return {"image": image, "loc_image": loc_image, 'contains_ship': 0}

    def _get_image_mask(self, idx):
        if pd.isnull(self.ships_dataframe.iloc[idx, 1]):
            ship_loc_data = []
        else:
            ship_loc_data = self.ships_dataframe.iloc[idx, 1]
            ship_loc_data = ship_loc_data.split(' ')
            ship_loc_data = np.asarray(ship_loc_data).reshape(-1, 2)

        loc_image = rle_decode(ship_loc_data, original_width, original_height)
        return loc_image

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache.get(idx)
        try:
            image_name = self.merge_ships_dataframe.iloc[idx, 0]
            image_idxs = self.merge_ships_dict[image_name]
            image_path = 'data/train_v2/' + image_name
            image = io.imread(image_path)
            loc_image = self._merge_images(image_idxs)
            sample = {'image': image, 'loc_image': loc_image}
        except:
            sample = self._get_blank_sample()
        sample = self.transform(sample)
        if self.use_cache:
            self.cache[idx] = sample
        return sample

    def _merge_ships(self, df):
        ships = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            image_name = row[0]
            if image_name in ships:
                ships[image_name].append(idx)
            else:
                ships[image_name] = [idx]
        return ships

    def _merge_images(self, idxs):
        img = rle_decode([], original_width, original_height)
        for idx in idxs:
            img = np.logical_or(img, self._get_image_mask(idx)).astype(float)

        return img.astype(np.uint8) * 255


class SubmissionToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if self.dimensions:
            image= self.resize_image(image)

        image = self.transpose_and_scale_image(image)
        return {'image': torch.from_numpy(image), 'image_name': sample['image_name']}

    @staticmethod
    def transpose_and_scale_image(image):
        return image.transpose((2, 0, 1)).astype(np.float32) / 255

    def resize_image(self, image):
        height, width = self.dimensions
        image = transform.resize(image, (height, width), anti_aliasing=True, preserve_range=True)
        return image

    @staticmethod
    def binarize_mask(loc_image):
        loc_image[loc_image > 0] = 255


class SubmissionShipData(Dataset):
    def __init__(self, image_width, image_height, use_cache=True, dropna=True):
        self.image_width = image_width
        self.image_height = image_height
        self.use_cache = use_cache
        self.cache = {}
        self.ships_dataframe = pd.read_csv('data/sample_submission_v2.csv')
        self.transform = transforms.Compose([SubmissionToTensor((image_height, image_width))])

    def __len__(self):
        return len(self.ships_dataframe) - 1

    def _get_blank_sample(self):
        image = np.zeros((self.image_height, self.image_width, 3))
        loc_image = rle_decode([], original_width, original_height)
        return {"image": image, "loc_image": loc_image, 'contains_ship': 0}

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache.get(idx)

        image_name = self.ships_dataframe.iloc[idx, 0]
        image_path = 'data/test_v2/' + image_name
        image = io.imread(image_path)
        sample = {'image': image, 'image_name': image_name}

        sample = self.transform(sample)
        if self.use_cache:
            self.cache[idx] = sample
        return sample
