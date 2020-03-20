import numpy as np
import tensorflow as tf
from secgan.models import ConvStack3DEncoder
from ffn.training.models import convstack_3d


class SwappedEncoderFFNModel(convstack_3d.ConvStack3DFFNModel):
    def __init__(
        self,
        encoder_ckpt,
        fov_size=None,
        deltas=None,
        batch_size=None,
        depth=9,
        layer=3,
    ):
        super().__init__(
            fov_size=fov_size, deltas=deltas, batch_size=batch_size, depth=depth
        )
        self.encoder_ckpt = encoder_ckpt
        self.layer = layer
        self.fov_size = fov_size
        self.get_encoder_vars()

    def define_tf_graph(self):
        super().define_tf_graph()

        # Add op to swap encoder weights from encoder ckpt
        # Caller needs to make sure this is run after initial restore
        einit_op, einit_fd = tf.contrib.framework.assign_from_values(
            self.encoder_var_names_to_values
        )

        self.init_op = einit_op
        self.init_feed_dict = einit_fd

    def get_encoder_vars(self):
        '''Load encoder weights and remap their names to ffn naming'''
        encoder_loading_graph = tf.Graph()
        with encoder_loading_graph.as_default():
            encoder = ConvStack3DEncoder(
                fov_size=self.fov_size,
                input_seed=np.zeros(
                    [self.batch_size, *self.fov_size, 1], dtype=np.float32
                ),
                batch_size=self.batch_size,
                depth=self.layer,
            )
            encoder.define_tf_graph()

            # Encoder var naming -> FFN var naming
            names = [
                v.op.name.replace('encoder', 'seed_update')
                for v in encoder.vars
            ]

            with tf.Session() as sess:
                encoder.saver.restore(sess, self.encoder_ckpt)
                weights = dict(zip(names, sess.run(encoder.vars)))

        self.encoder_var_names_to_values = weights
