import tensorflow as tf
from ffn.training.models import convstack_3d


class SwappedEncoderFFN(convstack_3d.ConvStack3DFFNModel):

    def __init__(
        self,
        encoder_ckpt,
        fov_size=None,
        deltas=None,
        batch_size=None,
        depth=9,
    ):
        super().__init__(
            fov_size=fov_size,
            deltas=deltas,
            batch_size=batch_size,
            depth=depth,
        )
        self.encoder_ckpt = encoder_ckpt

    def define_tf_graph(self):
        super().define_tf_graph()

        # Add op to swap encoder weights from encoder ckpt
        # Caller needs to make sure this is run after initial restore
        einit_op, einit_fd = tf.contrib.framework.assign_from_checkpoint(
            self.encoder_ckpt, self.encoder.vars, ignore_missing_vars=True
        )

        self.init_op = einit_op
        self.init_feed_dict = einit_fd
