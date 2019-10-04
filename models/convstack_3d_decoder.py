import tensorflow as tf
from ffn.training import optimizer


def _convstack_3d(net, depth=9):
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


class ConvStack3DDecoder:
    def __init__(
        self,
        fov_size=None,
        batch_size=None,
        loss_lambda=None,
        depth=9,
        define_global_step=True,
    ):
        self.depth = depth
        self.input_shape = [batch_size, *fov_size]
        self.input_encoding = tf.placeholder(
            tf.float32, shape=self.input_shape, name='encoding'
        )
        self.target = tf.placeholder(
            tf.float32, shape=self.input_shape, name='target'
        )
        self.encoding_loss_lambda = loss_lambda
        self.loss = None
        if define_global_step:
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False
            )

    def set_up_loss(self, encoder):
        pixel_mse = tf.reduce_mean(
            tf.squared_difference(self.decoding, self.target)
        )
        encoding_mse = tf.reduce_mean(
            tf.squared_difference(encoder.encode(self.decoding)),
            self.input_encoding,
        )
        loss = self.encoding_loss_lambda * encoding_mse + pixel_mse
        self.loss = tf.verify_tensor_all_finite(loss, 'Invalid loss detected')

    def set_up_optimizer(self, loss=None, max_gradient_entry_mag=0.7):
        self.opt = opt = optimizer.optimizer_from_flags()
        grads_and_vars = opt.compute_gradients(loss)

        for g, v in grads_and_vars:
            if g is None:
                tf.logging.error('Gradient is None: %s', v.op.name)

        if max_gradient_entry_mag > 0.0:
            grads_and_vars = [
                (
                    tf.clip_by_value(
                        g, -max_gradient_entry_mag, max_gradient_entry_mag
                    ),
                    v,
                )
                for g, v in grads_and_vars
            ]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(
                grads_and_vars,
                global_step=self.global_step,
                name='train_decoder',
            )

    def define_tf_graph(self, encoder):
        net = tf.expand_dims(self.input_encoding, axis=4)

        with tf.variable_scope('decoding', reuse=False):
            logits = _convstack_3d(net, self.depth)

        self.decoding = logits

        self.set_up_loss(encoder)
        self.set_up_optimizer()

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
