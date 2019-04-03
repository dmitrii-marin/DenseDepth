import keras.backend as K
import tensorflow as tf
from utils import nus_upscale

def interpolate(y_pred, size):
    data, locations = tf.split(y_pred, [1, 2], -1)

    def _inter_impl(input):
        res = tf.py_func(nus_upscale, [input[0], input[1], size], data.dtype)
        res.set_shape([None, None, None])
        return res

    res = tf.map_fn(
        _inter_impl,
        (locations, data),
        dtype=data.dtype
    )
    return res


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    has_locations = tf.equal(tf.shape(y_pred)[3], 3)
    y_pred = tf.cond(
        has_locations,
        lambda: interpolate(y_pred, tf.shape(y_true)[1:3]),
        lambda: y_pred
    )

    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


def noisy_depth_loss_function(y_true, y_pred, alpha=0.5, maxDepthVal=1000.0/10.0):
    has_locations = tf.equal(tf.shape(y_pred)[3], 3)
    y_pred = tf.cond(
        has_locations,
        lambda: interpolate(y_pred, tf.shape(y_true)[1:3]),
        lambda: y_pred
    )

    good = K.not_equal(y_true, 0.0)
    y_true = K.log(y_true)
    cnt = tf.to_float(tf.maximum(tf.count_nonzero(good, [1, 2]), 1))
    diff = tf.where(good, y_pred - y_true, tf.zeros_like(y_pred))

    l_depth = tf.reduce_sum(diff ** 2, [1, 2]) / cnt \
        - tf.reduce_sum(diff, [1, 2]) ** 2 / cnt ** 2

    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

    l_edges = K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)
    bad = tf.logical_or(tf.is_nan(l_edges), tf.is_inf(l_edges ))
    l_edges = tf.where(bad, tf.zeros_like(l_edges), l_edges)

    l_depth = K.mean(l_depth)
    l_edges = K.mean(l_edges) * alpha

    return l_depth + l_edges
