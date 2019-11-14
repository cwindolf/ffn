# import tensorflow as tf


def center_crop_vol(x, crop):
    return x[:, crop:-crop, crop:-crop, crop:-crop, :]
