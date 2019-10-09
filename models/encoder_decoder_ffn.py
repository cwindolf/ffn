import tensorflow as tf
from ffn.training import model
from training import inputs
from models import convstacktools, ConvStack3DDecoder, ConvStack3DEncoder
from ffn.training.models import convstack_3d


class EncoderDecoderFFN(model.FFNModel):
    '''An FFNModel with an encoder-decoder stuck on the beginning.'''

    dim = 3

    def __init__(
        self,
        weights=None,
        fov_size=None,
        batch_size=None,
        layer=None,
        decoder_ckpt=None,
        seed_pad=0.5,
        seed_init=0.95,
        depth=9,
        ffn_deltas=None,
    ):
        super().__init__(ffn_deltas, batch_size)
        self.batch_size = batch_size
        self.depth = depth
        self.input_shape = [batch_size, *fov_size, 1]
        self.weights = weights
        self.decoder_ckpt = decoder_ckpt

        # Init encoder and decoder
        self.encoder = ConvStack3DEncoder(
            weights=weights,
            fov_size=fov_size,
            batch_size=batch_size,
            input_seed=inputs.fixed_seed_batch(
                batch_size, fov_size, seed_pad, seed_init
            ),
            depth=layer,
        )
        self.decoder = ConvStack3DDecoder(
            fov_size=fov_size,
            batch_size=batch_size,
            depth=layer,
            define_global_step=False,
        )

        self.set_uniform_io_size(fov_size)
        # self.input_patches = None
        # self.input_seed = None

        self.saver = None

    def define_tf_graph(self):
        self.encoder.define_tf_graph()
        self.decoder.define_tf_graph(self.encoder)

        encoder_init_op = tf.initializers.variables(self.encoder.vars)
        decoder_init_op, self.init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
            self.decoder_ckpt,
            self.decoder.vars,
            ignore_missing_vars=True,
        )

        self.input_patches = self.encoder.input_patches

        print(self.decoder.decoding)
        print(self.input_seed)
        net = tf.concat([self.decoder.decoding, self.input_seed], 4)

        with tf.variable_scope('seed_update', reuse=False):
            logit_update = convstacktools.fixed_convstack_3d(
                net, self.weights, self.depth
            )

        my_init_op = tf.initializers.variables(
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='seed_update'
            )
        )
        self.init_op = tf.group(encoder_init_op, decoder_init_op, my_init_op)

        tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, self.init_op)

        logit_seed = self.update_seed(self.input_seed, logit_update)
        self.logits = logit_seed
        self.logistic = tf.sigmoid(logit_seed)

    @classmethod
    def from_ffn_ckpt(
        cls,
        ffn_ckpt,
        ffn_delta,
        fov_size,
        batch_size,
        decoder_ckpt,
        layer,
        depth=9
    ):
        ffn_deltas = [ffn_delta, ffn_delta, ffn_delta]
        ffn_loading_graph = tf.Graph()
        with ffn_loading_graph.as_default():
            ffn = convstack_3d.ConvStack3DFFNModel(
                fov_size=fov_size,
                deltas=ffn_deltas,
                batch_size=batch_size,
                depth=depth,
            )
            # Since ffn.labels == None, this will not set up training graph
            ffn.define_tf_graph()

            trainable_names = [v.op.name for v in tf.trainable_variables()]

            with tf.Session() as sess:
                ffn.saver.restore(sess, ffn_ckpt)
                weights = dict(
                    zip(trainable_names, sess.run(tf.trainable_variables()))
                )

        return cls(
            weights=weights,
            fov_size=fov_size,
            batch_size=batch_size,
            layer=layer,
            depth=depth,
            decoder_ckpt=decoder_ckpt,
            ffn_deltas=ffn_deltas,
        )
