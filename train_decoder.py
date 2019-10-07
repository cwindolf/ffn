import numpy as np
import os.path
import tensorflow as tf
from absl import app
from absl import flags
from training import inputs
from ffn.training.models import convstack_3d
import models


# ------------------------------- flags -------------------------------

# Model parameters
flags.DEFINE_integer(
    'layer',
    None,
    'Layer at which to truncate the FFN, and also the depth of the decoder.',
)
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_float('loss_lambda', 1e-3, 'Encoding loss coefficient')
flags.DEFINE_integer('fov_len', 33, 'Length of FOV on each axis')
flags.DEFINE_integer('ffn_delta', 8, '')
flags.DEFINE_float('seed_pad', 0.5, '')
flags.DEFINE_float('seed_init', 0.95, '')
flags.DEFINE_integer('depth', 9, 'Depth of original FFN model.')

# Data
flags.DEFINE_string(
    'volume_spec', None, 'Volume to encode and train decoder on.'
)
flags.DEFINE_float('image_mean', 128.0, '')
flags.DEFINE_float('image_stddev', 33.0, '')

# Model storage
flags.DEFINE_string('train_dir', None, 'Where to save decoder checkpoints.')
flags.DEFINE_string('ffn_ckpt', None, 'Load this up as the encoder.')


FLAGS = flags.FLAGS


# ------------------------------- main --------------------------------


def main(argv):
    # Parse args a little -----------------------------------------
    fov_size = [FLAGS.fov_len, FLAGS.fov_len, FLAGS.fov_len]
    ffn_deltas = [FLAGS.ffn_delta, FLAGS.ffn_delta, FLAGS.ffn_delta]

    with open(os.path.join(FLAGS.train_dir, 'flagfile.txt'), 'w') as ff:
        ff.write(FLAGS.flags_into_string())

    # Data pipeline -----------------------------------------------
    fov_batches = inputs.random_fovs(
        FLAGS.volume_spec,
        FLAGS.batch_size,
        fov_size,
        FLAGS.image_mean,
        FLAGS.image_stddev,
    )

    # Make a batch of "init" seeds to feed the encoder.
    # logit ???
    fixed_seed = np.full(fov_size, FLAGS.seed_pad, dtype=np.float32)
    fov_center = tuple(list(np.array(fov_size) // 2))
    fixed_seed[fov_center] = FLAGS.seed_init
    fixed_seed_batch = np.array([fixed_seed] * FLAGS.batch_size)[..., None]

    # Load FFN weights ------------------------------------------------
    # Hooking graphs together... This bit loads up weights.
    encoder_loading_graph = tf.Graph()
    with encoder_loading_graph.as_default():
        ffn = convstack_3d.ConvStack3DFFNModel(
            fov_size=fov_size,
            deltas=ffn_deltas,
            batch_size=FLAGS.batch_size,
            depth=FLAGS.depth,
        )
        # Since ffn.labels == None, this will not set up training graph
        ffn.define_tf_graph()

        trainable_names = [v.op.name for v in tf.trainable_variables()]

        with tf.Session() as sess:
            ffn.saver.restore(sess, FLAGS.ffn_ckpt)
            weights = dict(
                zip(trainable_names, sess.run(tf.trainable_variables()))
            )

    # Decoder graph ---------------------------------------------------
    decoding_graph = tf.Graph()
    with decoding_graph.as_default():
        # Build encoder -----------------------------------------------
        encoder = models.ConvStack3DEncoder(
            weights=weights,
            fov_size=fov_size,
            batch_size=FLAGS.batch_size,
            depth=FLAGS.layer,
        )
        encoder.define_tf_graph()

        # Build decoder -----------------------------------------------
        decoder = models.ConvStack3DDecoder(
            fov_size=fov_size,
            batch_size=FLAGS.batch_size,
            loss_lambda=FLAGS.loss_lambda,
            depth=FLAGS.layer,
            encoding=encoder.encoding,
        )
        decoder.define_tf_graph(encoder)

        # TF setup + run ----------------------------------------------
        scaffold = tf.train.Scaffold(
            ready_for_local_init_op=tf.report_uninitialized_variables(
                decoder.vars
            ),
            local_init_op=tf.group(
                [
                    tf.initializers.variables(encoder.vars),
                    tf.initializers.local_variables(),
                ]
            ),
            saver=decoder.saver,
            summary_op=tf.summary.merge_all(),
        )
        config = tf.ConfigProto(
            log_device_placement=False, allow_soft_placement=True
        )
        with tf.train.MonitoredTrainingSession(
            config=config,
            scaffold=scaffold,
            checkpoint_dir=FLAGS.train_dir,
            save_summaries_secs=300,
            save_checkpoint_secs=600,
        ) as sess:
            # Train decoder
            for fov_batch in fov_batches:
                # Run decoder train op
                sess.run(
                    decoder.train_op,
                    feed_dict={
                        encoder.input_patches: fov_batch,
                        encoder.input_seed: fixed_seed_batch,
                        decoder.target: fov_batch,
                    },
                )


# ---------------------------------------------------------------------
if __name__ == '__main__':
    flags.mark_flag_as_required('layer')
    flags.mark_flag_as_required('volume_spec')
    flags.mark_flag_as_required('train_dir')
    flags.mark_flag_as_required('ffn_ckpt')
    app.run(main)
