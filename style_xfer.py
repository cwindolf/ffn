import tensorflow as tf
import models
from absl import flags
from absl import app

# ------------------------------- flags -------------------------------

# Script args
flags.DEFINE_enum(
    'method', 'wc', ['wc', 'hm'], 'Whitening-coloring or histogram matching'
)
flags.DEFINE_string(
    'contentspec', None, 'Input datspec, the content source.')
flags.DEFINE_string(
    'stylespec', None, 'Target style datspec.')
flags.DEFINE_string(
    'outspec', None, 'Write output here.')


# Encoder flags
flags.DEFINE_string(
    'ffn_ckpt', None, 'If supplied, load FFN weights for the encoder.')
flags.DEFINE_string(
    'encoder_ckpt', None, 'If supplied, load encoder from a checkpoint.')

FLAGS = flags.FLAGS


# ------------------------------- lib ---------------------------------


def load_encoder(fov_size, seed):
    # Load FFN weights ------------------------------------------------
    # Hooking graphs together... This bit loads up weights.
    if FLAGS.ffn_ckpt:
        assert not FLAGS.encoder_ckpt
        encoder = models.ConvStack3DEncoder.from_ffn_ckpt(
            ffn_ckpt=FLAGS.ffn_ckpt,
            fov_size=fov_size,
            input_seed=seed,
            depth=FLAGS.layer,
        )

    # Training graph --------------------------------------------------
    encoder_graph = tf.Graph()
    with encoder_graph.as_default():
        # Init encoder ------------------------------------------------
        if not FLAGS.ffn_ckpt:
            assert FLAGS.encoder_ckpt
            encoder = models.ConvStack3DEncoder(
                input_seed=seed,
                fov_size=fov_size,
                depth=FLAGS.layer,
            )
        encoder.define_tf_graph()

        # Make init op
        if FLAGS.ffn_ckpt:
            einit_op = tf.initializers.variables(encoder.vars)

            def einit_fn(session):
                session.run(einit_op)
        else:
            def einit_fn(session):
                encoder.saver.restore(session, FLAGS.encoder_ckpt)

    return encoder, encoder_graph, einit_fn


# ------------------------------- main --------------------------------


def main(argv):
    # Load encoder ----------------------------------------------------
    encoder, encoder_graph, einit_fn = load_encoder()

    # Embed input volume ----------------------------------------------
    pass

    # Do feature transform --------------------------------------------
    pass

    # Load decoder ----------------------------------------------------
    pass

    # Decode transformed features -------------------------------------
    pass

    # Write to output volume ------------------------------------------
    pass


# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run(main)
