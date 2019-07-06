"""
Experiment to see if we can create a loc2vec as detailed in the blogpost.
bloglink: https://www.sentiance.com/2018/05/03/venue-mapping/
"""
import random
from pathlib import Path
# from collections import OrderedDict
# import time
import pandas as pd
import numpy as np

from PIL import Image
import torch
# from torch import nn, optim
from staticmap import StaticMap
from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
from torchvision import transforms

from lat2tile import deg2num
from tile_image import render_single_tile


def get_files_from_path(pathstring):
    """retrives file names from the folder and returns a pandas dataframe with
    four columns: path, filesize, lat, long

    Arguments:
        pathstring {string} -- relative location of file

    Returns:
        [pandas dataframe] -- sorted by the filesize
    """

    filenames = []
    for file in Path(pathstring).glob("**/*.png"):
        filenames.append((file, file.stat().st_size,
                          file.parts[-2], file.stem))
    files_df = pd.DataFrame(list(filenames),
                            columns=["path", "filesize", "x", "y"])
    sorted_files = files_df.sort_values("filesize")
    result_df = sorted_files.reset_index(drop=True)
    return result_df


def cleanse_files(df_files):
    """
    lets check filesizes and remove known useless tiles.

    103, 306, 355, 2165, 2146, 2128, 2202 are heavily
    represented and are typically grasslands/ empty / sea.

    Let's remove that from the samples!

    Arguments:
        df_files {pandas dataframe} -- should contain a column named "filesize"

    Returns:
        dataframe -- filtered dataframe with useless file sizes removed
    """

    filtered_files = df_files[(df_files["filesize"] != 103) &
                              (df_files["filesize"] != 306) &
                              (df_files["filesize"] != 355) &
                              (df_files["filesize"] != 2146) &
                              (df_files["filesize"] != 2128) &
                              (df_files["filesize"] != 2165) &
                              (df_files["filesize"] != 2202)]
    result = filtered_files.reset_index(drop=True)
    count = result.filesize.value_counts()
    freq = 1./count
    freq_dict = freq.to_dict()
    result['frequency'] = result['filesize'].map(freq_dict)
    print(len(result))
    return result




class GeoTileDataset(Dataset):
    """
    A custom dataset to provide a batch of geotiles.
    """

    transform = None
    center_transform = None
    ten_crop = None
    pd = None

    def __init__(self, path, transform, center_transform):
        # for us west
        zoom, self.startx, self.starty = deg2num(14, 48.995395, -124.866258)
        zoom, self.endx, self.endy = deg2num(14, 31.74734, -104.052404)
        self.SAMPLE_NUM = 50_000

        self.df_files = get_files_from_path(path)
        self.df_filtered_files = cleanse_files(self.df_files)
        self.m = StaticMap(500, 500, url_template='http://localhost:8080/tile/{z}/{x}/{y}.png')

        self.ten_crop = transforms.Compose([transforms.TenCrop(128)])
        self.transform = transform
        self.center_transform = center_transform

    def __getitem__(self, index):
        x, y = random.randint(self.startx, self.endx), random.randint(self.starty, self.endy)


        # random spot in the area, only renders when needed
        rand_lat = random.randrange(31.74734, 48.995395, 0.01)
        rand_lon = random.randrange(-124.866258, -104.052404, 0.01)
        print(rand_lat, rand_lon)
        ext = [rand_lat, rand_lon, rand_lat + 0.001, rand_lon + 0.001]

        data = render_single_tile(self.m, ext)

        # data = Image.open(self.df_filtered_files.iloc[index].path, 'r')
        data = data.convert('RGB')
        cropped_data = self.ten_crop(data)
        center_data_tensor = torch.stack([self.center_transform(data)
                                           for i in range(0,10)], 0)
        ten_data = torch.stack([self.transform(x) for x in cropped_data], 0)
        twenty_data = torch.cat([center_data_tensor, ten_data], 0)

        array_size = twenty_data.shape[0]
        tile_ids = torch.from_numpy((index)*np.ones([array_size, 1]))
        tile_ids = tile_ids.type(torch.long)
        return twenty_data, tile_ids

    def __len__(self):
        return self.SAMPLE_NUM

    def get_file_df(self):
        return self.df_filtered_files


class GeoTileInferDataset(Dataset):
    """
    A custom dataset to provide a single center cropped tile.
    """

    transform = None
    center_transform = None
    ten_crop = None
    pd = None

    def __init__(self, path, center_transform):
        self.df_files = get_files_from_path(path)
        self.df_filtered_files = cleanse_files(self.df_files)
        self.center_transform = center_transform

    def __getitem__(self, index):
        data = Image.open(self.df_filtered_files.iloc[index].path, 'r')
        data = data.convert('RGB')
        center_data = self.center_transform(data)
        return center_data, index

    def __len__(self):
        return self.df_filtered_files.shape[0]

    def get_file_df(self):
        return self.df_filtered_files