import numpy as np
from utils import DepthNorm, sampling
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy
import os, scipy, pickle, h5py

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_nyu_data(batch_size, nyu_data_zipfile='nyu_data.zip'):
    data = extract_zip(nyu_data_zipfile)

    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Helpful for testing...
    if False:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth

def get_nyu_train_test_data(batch_size, nus=None):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, nus=nus)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, nus=nus)

    return train_generator, test_generator

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, nus=None):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)
        self.N = len(self.dataset)
        self.nus = nus

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(480,640,3)/255,0,1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(480,640,1)/255*self.maxDepth,0,self.maxDepth)

            if self.nus is not None:
                u, v = sampling(y[:,:,0], self.shape_rgb[1:3], self.nus)
                x = x[v.astype(int), u.astype(int), :]
                u, v = sampling(y[:,:,0], self.shape_depth[1:3], self.nus)
                y = y[v.astype(int), u.astype(int), :]
            else:
                x = nyu_resize(x, 480)
                y = nyu_resize(y, 240)

            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy:
                batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth, nus=None):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.locations = True
        self.nus = nus

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        shape_rgb = self.shape_rgb
        shape_depth = self.shape_depth
        batch_x = np.zeros( shape_rgb[:3] + (5 if self.locations else 3,), np.float32 )
        if self.locations:
            batch_y = [] # size can vary
        else:
            batch_y = np.zeros( shape_depth[:3] + (1,), np.float32 )

        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(480,640,3)/255,0,1)
            y = np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0
            y = DepthNorm(y, maxDepth=self.maxDepth)

            # adjust sizes
            if self.nus is not None:
                u, v = sampling(y, self.shape_rgb[1:3], self.nus)
                locations = [v, u]
                x = x[v.astype(int), u.astype(int), :]
                if not self.locations:
                    u, v = sampling(y, self.shape_depth[1:3], self.nus)
                    y = y[v.astype(int), u.astype(int), :]
            else:
                shape = x.shape
                locations = np.mgrid[:shape[0]-1:shape_rgb[1]*1j, :shape[1]-1:shape_rgb[2]*1j]
                sample = np.round(locations).astype(int)
                x = x[sample[0], sample[1], :]
                if not self.locations:
                    sample = np.round(np.mgrid[:shape[0]-1:shape_depth[1]*1j, :shape[1]-1:shape_depth[2]*1j]).astype(int)
                    y = y[sample[0], sample[1]]

            batch_x[i, ..., :3] = x
            if self.locations:
                batch_x[i, ..., 3:] = np.stack(locations, -1)
                batch_y.append(y)
            else:
                batch_y[i, ..., 0] = y

        if self.locations:
            max_x, max_y = np.max([x.shape[:2] for x in batch_y], axis=0)
            batch_y = np.stack([
                np.pad(x, [(0, max_x - x.shape[0]), (0, max_y - x.shape[1]), (0, 0)], 'constant')
                for x in batch_y
            ])

        return batch_x, batch_y

#================
# Unreal dataset
#================

import cv2
from skimage.transform import resize

def get_unreal_data(batch_size, unreal_data_file='unreal_data.h5'):
    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Open data file
    import h5py
    data = h5py.File(unreal_data_file, 'r')

    # Shuffle
    from sklearn.utils import shuffle
    keys = shuffle(list(data['x'].keys()), random_state=0)

    # Split some validation
    unreal_train = keys[:len(keys)-100]
    unreal_test = keys[len(keys)-100:]

    # Helpful for testing...
    if False:
        unreal_train = unreal_train[:10]
        unreal_test = unreal_test[:10]

    return data, unreal_train, unreal_test, shape_rgb, shape_depth

def get_unreal_train_test_data(batch_size):
    data, unreal_train, unreal_test, shape_rgb, shape_depth = get_unreal_data(batch_size)

    train_generator = Unreal_BasicAugmentRGBSequence(data, unreal_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = Unreal_BasicAugmentRGBSequence(data, unreal_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, is_skip_policy=True)

    return train_generator, test_generator

class Unreal_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.N = len(self.dataset)
        self.is_skip_policy = is_skip_policy

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            rgb_sample = cv2.imdecode(np.asarray(self.data['x/{}'.format(sample)]), 1)
            depth_sample = self.data['y/{}'.format(sample)]
            depth_sample = resize(depth_sample, (self.shape_depth[1], self.shape_depth[2]), preserve_range=True, mode='reflect', anti_aliasing=True )

            x = np.clip(rgb_sample/255, 0, 1)
            y = np.clip(depth_sample, 10, self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i, ..., :1] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i],self.maxDepth)/self.maxDepth,0,1), index, i)

        return batch_x, batch_y

def get_megadepth_train_test_data(batch_size,
                                  root='/data/MegaDepth/MegaDepth_v1',
                                  lists='/data/MegaDepth/final_list'):

    shape_rgb = (480, 640)
    shape_depth = (240, 320)

    train = os.path.join(lists, 'train_list')
    val = os.path.join(lists, 'val_list')
    train_generator = MD_BasicAugmentRGBSequence(root, train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = MD_BasicAugmentRGBSequence(root, val, locations=True, batch_size=batch_size, is_skip_policy=True, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

class MD_BasicAugmentRGBSequence(Sequence):
    def __init__(self, root, dataset, batch_size, shape_rgb, shape_depth, locations=False, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.root = root
        self.shape_rgb = [
            (min(shape_rgb), max(shape_rgb)),
            (max(shape_rgb), min(shape_rgb)),
        ]
        self.shape_depth = [
            (min(shape_depth), max(shape_depth)),
            (max(shape_depth), min(shape_depth)),
        ]
        self.dataset = [
            {
                datum: pickle.load(open(os.path.join(dataset, orientation, datum + '_MD.p'), 'rb'))
                for datum in ["imgs", "targets"]
            }
            for orientation in ["landscape", "portrait"]
        ]
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.N = [len(dataset["imgs"]) for dataset in self.dataset]
        self.is_skip_policy = is_skip_policy
        self.locations = locations

    def __len__(self):
        return sum(int(np.ceil(N / float(self.batch_size))) for N in self.N)

    def __getitem__(self, idx, is_apply_policy=True):
        start = idx * self.batch_size

        for N, dataset, shape_rgb, shape_depth in zip(self.N, self.dataset, self.shape_rgb, self.shape_depth):
            if start < N:
                break
            start -= N

        batch_x = np.zeros( (self.batch_size,) + shape_rgb + (5 if self.locations else 3,), np.float32 )
        if self.locations:
            batch_y = [] # size can vary
        else:
            batch_y = np.zeros( (self.batch_size,) + shape_depth + (1,), np.float32 )

        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(self.batch_size):
            index = (start + i) % N

            with open(os.path.join(self.root, dataset["imgs"][index]), 'rb') as f:
                rgb_sample = np.array(Image.open(f).convert("RGB"), np.float32)
            with h5py.File(os.path.join(self.root, dataset["targets"][index]), 'r') as f:
                depth_sample = np.asarray(f["depth"])

            if np.min(depth_sample) < 0: # unsupported order data
                depth_sample[...] = 0.0 # mark unknown


            # adjust sizes
            shape = rgb_sample.shape
            if (shape_rgb[0] < shape_rgb[1]) != (shape[0] < shape[1]):
                 print("Orientation should not change! %s vs %s" % (shape_rgb, shape))

            locations = np.mgrid[:shape[0]-1:shape_rgb[0]*1j, :shape[1]-1:shape_rgb[1]*1j]
            sample = np.round(locations).astype(int)
            rgb_sample = rgb_sample[sample[0], sample[1], :]
            if not self.locations:
                sample = np.round(np.mgrid[:shape[0]-1:shape_depth[0]*1j, :shape[1]-1:shape_depth[1]*1j]).astype(int)
                depth_sample = depth_sample[sample[0], sample[1]]

            x = np.clip(rgb_sample/255, 0, 1)
            y = depth_sample

            if is_apply_policy:
                x, y = self.policy(x, y)

            batch_x[i, ..., :3] = x
            if self.locations:
                batch_x[i, ..., 3:] = np.stack(locations, -1)
                batch_y.append(y)
            else:
                batch_y[i, ..., 0] = y

        if self.locations:
            max_x, max_y = np.max([x.shape for x in batch_y], axis=0)
            batch_y = np.stack([
                np.pad(x, [(0, max_x - x.shape[0]), (0, max_y - x.shape[1])], 'constant')
                for x in batch_y
            ])[..., None]

        return batch_x, batch_y
