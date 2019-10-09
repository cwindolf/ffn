import tensorflow as tf


def convstack_3d(net, depth=9):
    conv = tf.contrib.layers.conv3d
    with tf.contrib.framework.arg_scope(
        [conv], num_outputs=32, kernel_size=(3, 3, 3), padding='SAME'
    ):
        net = conv(net, scope='conv0_a')
        net = conv(net, scope='conv0_b', activation_fn=None)

        for i in range(1, depth):
            with tf.name_scope('residual%d' % i):
                in_net = net
                net = tf.nn.relu(net)
                net = conv(net, scope='conv%d_a' % i)
                net = conv(net, scope='conv%d_b' % i, activation_fn=None)
                net += in_net

    net = tf.nn.relu(net)
    logits = conv(net, 1, (1, 1, 1), activation_fn=None, scope='conv_decoding')

    return logits


def fixed_convstack_3d(net, weights, depth=9):
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


    net = tf.nn.relu(net)
    logits = conv(net, 1, (1, 1, 1), activation_fn=None, scope='conv_lom', trainable=False,
        weights_initializer=tf.constant_initializer(weights[f'seed_update/conv_lom/weights']),
        biases_initializer=tf.constant_initializer(weights[f'seed_update/conv_lom/biases']))

    return logits