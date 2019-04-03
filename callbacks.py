import io
import random
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from utils import DepthNorm, predict, evaluate, nus_upscale

import tensorflow as tf

def make_image(tensor):
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)


def default_normalizer(y, minDepth=10, maxDepth=1000):
    return np.clip(DepthNorm(y, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def adjust_prediction(pred, gt):
    mask = gt > 0
    if np.count_nonzero(mask) > 0:
        pred = pred * np.exp(-np.mean(np.log(pred[mask] / gt[mask])))
    return pred


def get_nyu_callbacks(model, basemodel, train_generator, test_generator,
                      test_set, runPath, depth_norm=default_normalizer,
                      get_depth=lambda x:x):
    callbacks = []

    # Callback: Tensorboard
    class LRTensorBoard(keras.callbacks.TensorBoard):
        def __init__(self, log_dir, epoch_size, *args, **kwargs):
            super().__init__(log_dir=log_dir, *args, **kwargs)

            self.num_samples = 6
            self.train_idx = np.random.randint(low=0, high=len(train_generator), size=10)
            self.test_idx = np.random.randint(low=0, high=len(test_generator), size=10)
            self.epoch_size = epoch_size

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            step = batch + self.epoch * self.epoch_size
            if batch % 100 == 0:
                self.update_info(step, logs)
            super().on_batch_end(batch, logs)

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            super().on_epoch_begin(epoch, logs)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            step = self.epoch * self.epoch_size
            self.update_info(step, logs)
            super().on_epoch_end(epoch, logs)

        def update_info(self, step, logs=None):
            # Samples using current model
            import matplotlib.pyplot as plt
            from skimage.transform import resize
            plasma = plt.get_cmap('plasma')

            train_samples = []
            test_samples = []

            for i in range(self.num_samples):
                x_train, y_train = train_generator.__getitem__(self.train_idx[i], False)
                x_test, y_test = test_generator[self.test_idx[i]]

                x_train, y_train = x_train[0], y_train[0, ..., 0]
                x_test, y_test = x_test[0], y_test[0, ..., 0]

                predict_train = get_depth(model.predict(x_train[None,...]))[0,:,:,0]
                predict_test = get_depth(model.predict(x_test[None,...]))[0,:,:,0]

                y_train = resize(y_train, predict_train.shape, preserve_range=True, mode='reflect', anti_aliasing=True)
                y_test = resize(y_test, predict_test.shape, preserve_range=True, mode='reflect', anti_aliasing=True)

                np.save('a%d.npy'%i, predict_train)
                np.save('b%d.npy'%i, y_train)
                predict_train = adjust_prediction(predict_train, y_train)
                predict_test = adjust_prediction(predict_test, y_test)

                min = np.min([np.min(predict_train), np.min(y_train)])
                max = np.max([np.max(predict_train), np.max(y_train)])

                predict_train = plasma((predict_train - min) / (max - min))[..., :3]
                gt_train = plasma((y_train - min) / (max - min))[..., :3]

                min = np.min([np.min(predict_test), np.min(y_test)])
                max = np.max([np.max(predict_test), np.max(y_test)])

                predict_test  = plasma((predict_test - min) / (max - min)) [..., :3]
                gt_test  = plasma((y_test - min) / (max - min)) [..., :3]

                rgb_train = resize(x_train[..., :3], predict_train.shape[:2], preserve_range=True, mode='reflect', anti_aliasing=True)
                rgb_test = resize(x_test[..., :3], predict_test.shape[:2], preserve_range=True, mode='reflect', anti_aliasing=True)

                train_samples.append(np.vstack([rgb_train, gt_train, predict_train]))
                test_samples.append(np.vstack([rgb_test, gt_test, predict_test]))

            for samples in [train_samples, test_samples]:
                for idx in range(len(samples)):
                    samples[idx] = resize(samples[idx], samples[0].shape[:2], preserve_range=True, mode='reflect', anti_aliasing=True)

            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_image(255 * np.hstack(train_samples)))]), step)
            self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_image(255 * np.hstack(test_samples)))]), step)

            if not test_set == None:
                logs = logs or {}
                # Metrics
                e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True)
                logs.update({'rel': e[3]})
                logs.update({'rms': e[4]})
                logs.update({'log10': e[5]})

    callbacks.append( LRTensorBoard(log_dir=runPath, epoch_size=len(train_generator), update_freq=500) )

    # Callback: Learning Rate Scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00009, min_delta=1e-2)
    callbacks.append( lr_schedule ) # reduce learning rate when stuck

    # Callback: save checkpoints
    callbacks.append(keras.callbacks.ModelCheckpoint(runPath + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
        verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=1))

    return callbacks
