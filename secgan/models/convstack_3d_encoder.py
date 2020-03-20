import tensorflow as tf
from ffn.training import optimizer
from ffn.training.models import convstack_3d
from secgan.models import convstacktools


class ConvStack3DEncoder:
    '''Meant to load up an FFN checkpoint and hold it fixed'''

    def __init__(
        self,
        weights=None,
        fov_size=None,
        input_seed=None,
        batch_size=1,
        for_training=False,
        encoding_loss_lambda=1.0,
        pixel_loss_lambda=1.0,
        depth=9,
        seed_as_placeholder=False,
    ):
        self.batch_size = batch_size
        self.depth = depth
        self.half_fov_z = fov_size[0] // 2
        self.input_shape = [batch_size, *fov_size, 1]
        self.weights = weights
        self.pixel_loss_lambda = pixel_loss_lambda
        self.encoding_loss_lambda = encoding_loss_lambda
        self.vars = []
        self.input_patches = None
        self.input_seed = input_seed
        self.input_patches_and_seed = None
        self.global_step = None
        self.for_training = for_training
        self.seed_as_placeholder = seed_as_placeholder

    def set_up_loss(self, decoder):
        decoding = decoder.decode(self.encoding)
        pixel_mse = tf.reduce_mean(
            tf.squared_difference(self.input_patches, decoding)
        )
        reencoding = self.encode(decoding)
        encoding_mse = tf.reduce_mean(
            tf.squared_difference(reencoding, self.encoding)
        )
        loss = self.pixel_loss_lambda * pixel_mse + encoding_mse
        self.loss = tf.verify_tensor_all_finite(loss, 'Invalid loss detected')

        # Some summaries
        tf.summary.scalar('encoder_metrics/pixel_mse', pixel_mse)
        tf.summary.scalar('encoder_metrics/encoding_mse', encoding_mse)
        tf.summary.scalar('encoder_metrics/loss', loss)
        tf.summary.image(
            'encoder/orig_and_decoded_encoding',
            tf.concat(
                [
                    self.input_patches[:, self.half_fov_z, ...],
                    decoding[:, self.half_fov_z, ...],
                ],
                axis=1,
            ),
        )
        tf.summary.image(
            'encoder/encoding_and_reencoded_decoding',
            tf.concat(
                [
                    self.encoding[:, self.half_fov_z, ..., 0, None],
                    reencoding[:, self.half_fov_z, ..., 0, None],
                ],
                axis=1,
            ),
        )

    def set_up_optimizer(self, loss=None, max_gradient_entry_mag=0.7):
        if loss is None:
            loss = self.loss

        with tf.variable_scope('encoder', reuse=False):
            self.opt = opt = optimizer.optimizer_from_flags()
            tf.logging.info(opt)
            grads_and_vars = opt.compute_gradients(
                loss,
                var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder'
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
                    name='train_encoder',
                )

    def add_training_ops(self, decoder):
        assert self.for_training
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.vars += [self.global_step]
        self.set_up_loss(decoder)
        self.set_up_optimizer()

        self.vars += [
            v
            for v in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'
            )
            if v not in self.vars
        ]

        self.saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=1, var_list=self.vars
        )

    def define_tf_graph(self):
        self.input_patches = tf.placeholder(
            tf.float32, shape=self.input_shape, name='encoder_patches'
        )

        if self.seed_as_placeholder:
            self.input_seed = tf.placeholder(
                tf.float32, shape=self.input_shape, name='input_seed'
            )
        else:
            self.input_seed = tf.constant(self.input_seed)

        self.input_patches_and_seed = tf.concat(
            [self.input_patches, self.input_seed], axis=4
        )

        with tf.variable_scope('encoder', reuse=False):
            encoding = convstacktools.peeping_convstack_3d(
                self.input_patches_and_seed,
                weights=self.weights,
                trainable=self.for_training,
                depth=self.depth,
            )

        self.encoding = encoding

        self.vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'
        )

        if not self.for_training:
            self.saver = tf.train.Saver(
                keep_checkpoint_every_n_hours=1, var_list=self.vars
            )

    def encode(self, input_fov):
        '''Add ops to encode another FOV other than the input placeholder.'''
        input_fov_and_seed = tf.concat([input_fov, self.input_seed], axis=4)
        with tf.variable_scope('encoder', reuse=True):
            encoded_fov = convstacktools.peeping_convstack_3d(
                input_fov_and_seed,
                weights=self.weights,
                trainable=self.for_training,
                depth=self.depth,
            )

        return encoded_fov

    @classmethod
    def from_ffn_ckpt(
        cls,
        ffn_ckpt=None,
        ffn_delta=0,
        fov_size=None,
        batch_size=1,
        input_seed=None,
        encoding_loss_lambda=1.0,
        pixel_loss_lambda=1.0,
        for_training=False,
        depth=9,
        seed_as_placeholder=False,
    ):
        '''Load fixed-weight encoder by truncating FFN.'''
        ffn_deltas = [ffn_delta, ffn_delta, ffn_delta]
        encoder_loading_graph = tf.Graph()
        with encoder_loading_graph.as_default():
            ffn = convstack_3d.ConvStack3DFFNModel(
                fov_size=fov_size,
                deltas=ffn_deltas,
                batch_size=batch_size,
                depth=depth,
            )
            # Since ffn.labels == None, this will not set up training graph
            ffn.define_tf_graph()

            trainable_names = [v.op.name for v in tf.trainable_variables()]

            with tf.Session(graph=encoder_loading_graph) as sess:
                ffn.saver.restore(sess, ffn_ckpt)
                weights = dict(
                    zip(trainable_names, sess.run(tf.trainable_variables()))
                )

        return cls(
            weights=weights,
            fov_size=fov_size,
            batch_size=batch_size,
            input_seed=input_seed,
            encoding_loss_lambda=encoding_loss_lambda,
            pixel_loss_lambda=pixel_loss_lambda,
            for_training=for_training,
            seed_as_placeholder=seed_as_placeholder,
            depth=depth,
        )
