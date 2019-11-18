import logging
import tensorflow as tf


def convdisc(net, norm='instance', depth=3):
    logging.info('Build conv discriminator.')

    conv = tf.contrib.layers.conv3d
    if norm == 'batch':
        norm = tf.contrib.layers.batch_norm
    elif norm == 'instance':
        norm = tf.contrib.layers.instance_norm
    else:

        def norm(net, scope=None):
            return net

    # Conv layers
    num_outputs = 32
    for d in range(depth):
        net = conv(
            net,
            activation_fn=None,
            num_outputs=num_outputs,
            stride=(2, 2, 2),
            kernel_size=(5, 5, 5),
            scope=f'conv{d}',
        )
        net = norm(net, scope=f'norm{d}')
        net = tf.nn.relu(net)
        num_outputs *= 2
        logging.info(f' - Shape of conv{d}: {net.shape}')

    with tf.variable_scope('probs'):
        # total spatial average pooling
        net = tf.reduce_mean(net, axis=(1, 2, 3))
        logging.info(f' - Shape after avg pool: {net.shape}')
        # linear output layer
        probs = tf.contrib.layers.fully_connected(net, 1, activation_fn=None)

    logging.info(f' - Shape of probs: {probs.shape}')

    return probs


def resnet18(net, norm='instance', early_stride=2, early_maxpool=False):
    '''
    This proverbial saying probably arose from the pick-purse always
    seizing upon the prey nearest him: his maxim being that of the
    Pope's man of gallantry:
        "The thing at hand is of all things the best."

    Enter CHAMBERLAIN:

    The architecture of the discriminator followed that of ResNet-18,
    but we eliminated the finalclassification layer and the global
    average pooling preceding it, and used the output of the last
    residual module directly, which we found necessary for the
    network to train stably.

    Following original lua impl:
    github.com/facebookarchive/fb.resnet.torch/blob/master/models/resnet.lua
    and
    https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
    '''
    logging.info('Build ResNet-18')

    conv = tf.contrib.layers.conv3d
    if norm == 'batch':
        norm = tf.contrib.layers.batch_norm
    elif norm == 'instance':
        norm = tf.contrib.layers.instance_norm
    else:

        def norm(net, scope=None):
            return net

    # conv1
    with tf.variable_scope('conv1'):
        # Conv-BN-Relu
        net = conv(
            net,
            activation_fn=None,
            scope='conv1',
            num_outputs=64,
            stride=(early_stride, early_stride, early_stride),
            kernel_size=(7, 7, 7),
            padding='SAME',
        )
        net = norm(net, scope='norm')
        net = tf.nn.relu(net)

        if early_maxpool:
            # max pool 3x3 blocks, stride 2, zero pad 1 px
            net = tf.nn.max_pool3d(
                net,
                ksize=(1, 3, 3, 3, 1),
                strides=(1, 2, 2, 2, 1),
                padding='SAME',
            )

    logging.info(f' - Shape after conv1: {net.shape}')

    # basic res layer 64 outputs, 2 repeats, stride 1
    with tf.contrib.framework.arg_scope(
        [conv],
        num_outputs=64,
        kernel_size=(3, 3, 3),
        padding='SAME',
        stride=(1, 1, 1),
    ):
        for sublayer in (1, 2):
            with tf.variable_scope(f'conv2_{sublayer}'):
                shortcut = net
                net = conv(net, activation_fn=None, scope='conv1')
                net = norm(net, scope='norm1')
                net = tf.nn.relu(net)
                net = conv(net, activation_fn=None, scope='conv2')
                net = norm(net, scope='norm2')
                net = net + shortcut
                net = tf.nn.relu(net)

    logging.info(f' - Shape after conv2: {net.shape}')

    # basic res layer 128 outputs, 2 repeats, stride 2
    # basic res layer 256 outputs, 2 repeats, stride 2
    # basic res layer 512 outputs, 2 repeats, stride 2
    for layer, channels in enumerate((128, 256, 512), start=3):
        with tf.contrib.framework.arg_scope(
            [conv], num_outputs=channels, kernel_size=(3, 3, 3), padding='SAME'
        ):
            for sublayer in (1, 2):
                with tf.variable_scope(f'conv{layer}_{sublayer}'):
                    if sublayer == 1:
                        shortcut = conv(
                            net, kernel_size=(1, 1, 1), stride=(2, 2, 2)
                        )
                    elif sublayer == 2:
                        shortcut = net
                    net = conv(
                        net,
                        activation_fn=None,
                        scope='conv1',
                        stride=(2, 2, 2) if sublayer == 1 else (1, 1, 1),
                    )
                    net = norm(net, scope='norm1')
                    net = tf.nn.relu(net)
                    net = conv(
                        net,
                        activation_fn=None,
                        scope='conv2',
                        stride=(1, 1, 1),
                    )
                    net = norm(net, scope='norm2')
                    net = net + shortcut
                    net = tf.nn.relu(net)

            logging.info(f' - Shape after conv{layer}: {net.shape}')

    with tf.variable_scope('probs'):
        # total spatial average pooling
        net = tf.reduce_mean(net, axis=(1, 2, 3))

        logging.info(f' - Shape after avg pool: {net.shape}')

        # linear output layer
        probs = tf.contrib.layers.fully_connected(net, 1, activation_fn=None)

    logging.info(f' - Shape of probs: {probs.shape}')

    return probs
