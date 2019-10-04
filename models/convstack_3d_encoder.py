import tensorflow as tf


def _fixed_convstack_3d(net, depth=9):
    """Copy of _predict_object_mask to peep at features.

    This sets trainable=False to hold the model fixed."""
    conv = tf.contrib.layers.conv3d
    with tf.contrib.framework.arg_scope(
        [conv],
        num_outputs=32,
        kernel_size=(3, 3, 3),
        padding='SAME',
        trainable=False,
    ):
        net = conv(net, scope='conv0_a')
        conv_as = [net]
        net = conv(net, scope='conv0_b', activation_fn=None)

        for i in range(1, depth):
            with tf.name_scope('residual%d' % i):
                in_net = net
                net = tf.nn.relu(net)
                net = conv(net, scope='conv%d_a' % i)
                conv_as.append(net)
                net = conv(net, scope='conv%d_b' % i, activation_fn=None)
                net += in_net

    net = tf.nn.relu(net)
    logits = conv(
        net,
        num_outputs=1,
        kernel_size=(1, 1, 1),
        activation_fn=None,
        scope='conv_lom',
        trainable=False,
    )

    return logits, conv_as


class ConvStack3DEncoder:
    '''Meant to load up an FFN checkpoint and hold it fixed'''

    def __init__(self, layer, fov_size=None, batch_size=None, depth=9):
        self.batch_size = batch_size
        self.layer = layer
        self.depth = depth
        self.input_shape = [batch_size, *fov_size]
        self.input_patches = tf.placeholder(
            tf.float32, shape=self.input_shape, name='patches'
        )
        self.input_seed = tf.placeholder(
            tf.float32, shape=self.input_shape, name='seed'
        )
        self.input_patches_and_seed = tf.stack(
            [self.input_patches, self.input_seed], axis=4
        )

    def define_tf_graph(self):
        with tf.variable_scope('seed_update', reuse=False):
            self.logits, self.conv_as = _fixed_convstack_3d(
                self.input_patches_and_seed, depth=self.depth
            )
        self.encoding = self.conv_as[self.layer]
        self.saver = tf.train.Saver()

    def encode(self, input_fov):
        input_fov_and_seed = tf.stack(
            [input_fov, self.input_seed], axis=4
        )
        with tf.variable_scope('seed_update', reuse=True):
            _, conv_as = _fixed_convstack_3d(
                input_fov_and_seed, depth=self.depth
            )

        return conv_as[self.layer]
