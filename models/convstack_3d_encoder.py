import tensorflow as tf


def _fixed_convstack_3d(net, weights, depth=9):
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
        net = conv(
            net,
            scope='conv0_a',
            weights_initializer=tf.constant_initializer(
                weights['seed_update/conv0_a/weights']
            ),
            biases_initializer=tf.constant_initializer(
                weights['seed_update/conv0_a/biases']
            ),
        )
        net = conv(
            net,
            scope='conv0_b',
            activation_fn=None,
            weights_initializer=tf.constant_initializer(
                weights['seed_update/conv0_b/weights']
            ),
            biases_initializer=tf.constant_initializer(
                weights['seed_update/conv0_b/biases']
            ),
        )

        for i in range(1, depth):
            with tf.name_scope(f'residual{i}'):
                in_net = net
                net = tf.nn.relu(net)
                net = conv(
                    net,
                    scope=f'conv{i}_a',
                    weights_initializer=tf.constant_initializer(
                        weights[f'seed_update/conv{i}_a/weights']
                    ),
                    biases_initializer=tf.constant_initializer(
                        weights[f'seed_update/conv{i}_a/biases']
                    ),
                )

                if i == depth - 1:
                    return net

                net = conv(
                    net,
                    scope=f'conv{i}_b',
                    activation_fn=None,
                    weights_initializer=tf.constant_initializer(
                        weights[f'seed_update/conv{i}_b/weights']
                    ),
                    biases_initializer=tf.constant_initializer(
                        weights[f'seed_update/conv{i}_b/biases']
                    ),
                )
                net += in_net


class ConvStack3DEncoder:
    '''Meant to load up an FFN checkpoint and hold it fixed'''

    def __init__(
        self, ffn=None, weights=None, fov_size=None, batch_size=None, depth=9
    ):
        self.batch_size = batch_size
        self.depth = depth
        self.input_shape = [batch_size, *fov_size, 1]
        self.weights = weights
        self.input_patches = tf.placeholder(
            tf.float32, shape=self.input_shape, name='encoder_patches'
        )
        self.input_seed = tf.placeholder(
            tf.float32, shape=self.input_shape, name='encoder_seed'
        )
        self.input_patches_and_seed = tf.concat(
            [self.input_patches, self.input_seed], axis=4
        )
        self.vars = []

    def define_tf_graph(self):
        with tf.variable_scope('encode', reuse=False):
            encoding = _fixed_convstack_3d(
                self.input_patches_and_seed, self.weights, depth=self.depth
            )
            self.encoding = tf.stop_gradient(encoding)
        self.vars += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='encode'
        )

    def encode(self, input_fov):
        input_fov_and_seed = tf.concat([input_fov, self.input_seed], axis=4)
        with tf.variable_scope('encode_fov', reuse=False):
            encoded_fov = _fixed_convstack_3d(
                input_fov_and_seed, self.weights, depth=self.depth
            )
        self.vars += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='encode_fov'
        )

        return tf.stop_gradient(encoded_fov)
