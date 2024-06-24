import os.path as osp
import pickle
import random

import numpy as np
import torch

import lmdb

import numpy as np
from torch.utils.data import Dataset
import torchvision
import kornia
from skimage.color import lab2rgb


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


class BaseDataset(Dataset):
    def __init__(self, data_opt, **kwargs):
        # dict to attr
        for kw, args in data_opt.__dict__.items():
            setattr(self, kw, args)


    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

   
    @staticmethod
    def init_lmdb(seq_dir):
        env = lmdb.open(
            seq_dir, readonly=True, lock=False, readahead=False, meminit=False)
        return env

    @staticmethod
    def parse_lmdb_key(key):
        key_lst = key.split('_')
        idx, size, frm = key_lst[:-2], key_lst[-2], int(key_lst[-1])
        idx = '_'.join(idx)
        size = tuple(map(int, size.split('x')))  # n_frm, h, w
        return idx, size, frm

    @staticmethod
    def read_lmdb_frame(env, key, size):
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        frm = np.frombuffer(buf, dtype=np.uint8).reshape(*size)
        return frm

    def crop_sequence(self, **kwargs):
        pass

    @staticmethod
    def augment_sequence(**kwargs):
        pass
    
    
class PairedLMDBDataset(BaseDataset):
    """ LMDB dataset for paired data (for BI degradation)
    """

    def __init__(self, data_opt, **kwargs):
        super(PairedLMDBDataset, self).__init__(data_opt, **kwargs)
        
        self.lr_seq_dir = data_opt.lr_seq_dir
        self.data_type = data_opt.data_type
        self.scale = data_opt.scale
        self.tempo_extent = data_opt.tempo_extent
        self.moving_first_frame = data_opt.moving_first_frame
        self.moving_factor = data_opt.moving_factor
        self.filter_file = data_opt.filter_file
        
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256))])

        # load meta info
        lr_meta = pickle.load(
            open(osp.join(self.lr_seq_dir, 'meta_info.pkl'), 'rb'))
        self.lr_keys = sorted(lr_meta['keys'])
        
        # register parameters
        self.lr_env = None

    def __len__(self):
        return len(self.lr_keys)

    def __getitem__(self, item):
        if self.lr_env is None:
            self.lr_env = self.init_lmdb(self.lr_seq_dir)

        # parse info
        lr_key = self.lr_keys[item]
        idx, (tot_frm, lr_h, lr_w), cur_frm = self.parse_lmdb_key(lr_key)

        c = 3 if self.data_type.lower() == 'rgb' else 1
        
        # get frames
        lr_frms = []
        if self.moving_first_frame and (random.uniform(0, 1) > self.moving_factor):
            # load the first gt&lr frame
            lr_frm = self.read_lmdb_frame(
                self.lr_env, lr_key, size=(lr_h, lr_w, c))
            lr_frm = lr_frm.transpose(2, 0, 1)  # chw|rgb|uint8

            # generate random moving parameters
            offsets = np.floor(
                np.random.uniform(-1.5, 1.5, size=(self.tempo_extent, 2)))
            offsets = offsets.astype(np.int32)
            pos = np.cumsum(offsets, axis=0)
            min_pos = np.min(pos, axis=0)
            topleft_pos = pos - min_pos
            range_pos = np.max(pos, axis=0) - min_pos
            c_h, c_w = lr_h - range_pos[0], lr_w - range_pos[1]

            # generate frames
            for i in range(self.tempo_extent):
                lr_top, lr_left = topleft_pos[i]
                lr_frms.append(lr_frm[
                    :, lr_top: lr_top + c_h, lr_left: lr_left + c_w].copy())

        else:
            # read frames
            for i in range(cur_frm, cur_frm + self.tempo_extent):
                if i >= tot_frm:
                    # reflect temporal paddding, e.g., (0,1,2) -> (0,1,2,1,0)
                    lr_key = '{}_{}x{}x{}_{:04d}'.format(
                        idx, tot_frm, lr_h, lr_w, 2 * tot_frm - i - 2)
                else:
                    lr_key = '{}_{}x{}x{}_{:04d}'.format(
                        idx, tot_frm, lr_h, lr_w, i)

                lr_frm = self.read_lmdb_frame(
                    self.lr_env, lr_key, size=(lr_h, lr_w, c))
                lr_frm = lr_frm.transpose(2, 0, 1)
                lr_frms.append(lr_frm)

        lr_frms = np.stack(lr_frms)
        lr_tsr = torch.FloatTensor(np.ascontiguousarray(lr_frms)) / 255
        lr_tsr = self.transform(lr_tsr)

        lr_tsr = kornia.color.rgb_to_lab(lr_tsr)
        L = lr_tsr[:, 0:1, :, :]
        ab = lr_tsr[:, 1:, :, :]
        
        L = L/ 50. - 1. # Between -1 and 1
        ab = ab / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}

 
    @staticmethod
    def augment_sequence(gt_pats, lr_pats):
        # flip
        axis = random.randint(1, 3)
        if axis > 1:
            gt_pats = np.flip(gt_pats, axis)
            lr_pats = np.flip(lr_pats, axis)

        # rotate 90 degree
        k = random.randint(0, 3)
        gt_pats = np.rot90(gt_pats, k, (2, 3))
        lr_pats = np.rot90(lr_pats, k, (2, 3))

        return gt_pats, lr_pats
    
class DatasetConfig:
    def __init__(self, 
                 lr_seq_dir,
                 data_type='rgb',
                 scale=1,
                 tempo_extent=5,
                 moving_first_frame=False,
                 moving_factor=0.5,
                 filter_file=None):
        self.lr_seq_dir = lr_seq_dir
        self.data_type = data_type
        self.scale = scale
        self.tempo_extent = tempo_extent
        self.moving_first_frame = moving_first_frame
        self.moving_factor = moving_factor
        self.filter_file = filter_file

data_opt = DatasetConfig(
    lr_seq_dir='/media/moose/Moose/Dataset/AMD/',
    data_type='rgb',
    scale=1,
    tempo_extent=10,
    moving_first_frame=False,
    moving_factor=0.5,
    filter_file=None
)
