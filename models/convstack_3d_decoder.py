import tensorflow as tf
from ffn.training import optimizer
from models import convstacktools


class ConvStack3DDecoder:
    def __init__(
        self,
        fov_size=None,
        batch_size=None,
        loss_lambda=1e-3,
        depth=9,
        for_training=True,
    ):
        self.depth = depth
        self.half_fov_z = fov_size[0] // 2
        self.input_shape = [batch_size, *fov_size, 32]
        self.target_shape = [batch_size, *fov_size, 1]
        self.target = None
        self.encoding_loss_lambda = loss_lambda
        self.loss = None
        self.input_encoding = None
        self.for_training = for_training

    def set_up_loss(self, encoder):
        pixel_mse = tf.reduce_mean(
            tf.squared_difference(self.decoding, self.target)
        )
        reencoding = encoder.encode(self.decoding)
        encoding_mse = tf.reduce_mean(
            tf.squared_difference(reencoding, self.input_encoding)
        )
        loss = self.encoding_loss_lambda * encoding_mse + pixel_mse
        self.loss = tf.verify_tensor_all_finite(loss, 'Invalid loss detected')

        # Some summaries
        tf.summary.scalar('decoder_metrics/pixel_mse', pixel_mse)
        tf.summary.scalar('decoder_metrics/encoding_mse', encoding_mse)
        tf.summary.scalar('decoder_metrics/loss', loss)
        tf.summary.image(
            'decoder/orig_and_decoded_encoding',
            tf.concat(
                [
                    self.target[:, self.half_fov_z, ...],
                    self.decoding[:, self.half_fov_z, ...],
                ],
                axis=1,
            ),
        )
        tf.summary.image(
            'decoder/encoding_and_reencoded_decoding',
            tf.concat(
                [
                    self.input_encoding[:, self.half_fov_z, ..., 0, None],
                    reencoding[:, self.half_fov_z, ..., 0, None],
                ],
                axis=1,
            ),
        )

    def set_up_optimizer(self, loss=None, max_gradient_entry_mag=0.7):
        if loss is None:
            loss = self.loss

        with tf.variable_scope('decoder', reuse=False):
            self.opt = opt = optimizer.optimizer_from_flags()
            tf.logging.info(opt)
            grads_and_vars = opt.compute_gradients(
                loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder'
                ),
            )

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

    def decode(self, input_fov):
        with tf.variable_scope('decoder', reuse=True):
            decoded_fov = convstacktools.convstack_3d(
                input_fov, self.depth, trainable=self.for_training
            )

        return decoded_fov

    def define_tf_graph(self, encoder=None):
        if self.for_training:
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False
            )

        if encoder is not None:
            self.input_encoding = encoder.encoding
        else:
            self.input_encoding = tf.placeholder(
                tf.float32, shape=self.input_shape, name='input_encoding'
            )

        with tf.variable_scope('decoder', reuse=False):
            logits = convstacktools.convstack_3d(
                self.input_encoding, self.depth, trainable=self.for_training
            )

        self.decoding = logits

        self.vars = []

        if self.for_training:
            self.target = tf.placeholder(
                tf.float32, shape=self.target_shape, name='target'
            )
            self.set_up_loss(encoder)
            self.set_up_optimizer()

            self.vars += [self.global_step]

        self.vars += tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder'
        )

        self.saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=1, var_list=self.vars
        )
