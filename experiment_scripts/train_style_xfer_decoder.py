import os.path
import tensorflow as tf
from absl import app
from absl import flags
from secgan.training import inputs
from secgan import models
import logging


# ------------------------------- flags -------------------------------

# Model parameters
flags.DEFINE_integer(
    'layer',
    None,
    'Layer at which to truncate the FFN, and also the depth of the decoder.',
)
flags.DEFINE_integer('batch_size', 4, '')
flags.DEFINE_float('loss_lambda', 1e-3, 'Encoding loss coefficient')
flags.DEFINE_integer('fov_len', 33, 'Length of FOV on each axis')
flags.DEFINE_integer('ffn_delta', 8, '')
flags.DEFINE_float('seed_pad', 0.5, '')
flags.DEFINE_float('seed_init', 0.95, '')
flags.DEFINE_integer('depth', 12, 'Depth of original FFN model.')

# Data
flags.DEFINE_string(
    'volume_spec', None, 'Volume to encode and train decoder on.'
)
flags.DEFINE_float('image_mean', 128.0, '')
flags.DEFINE_float('image_stddev', 33.0, '')
flags.DEFINE_bool(
    'flat_seed', True, 'Use uniform prior seed instead of eyeball.'
)

# Model storage
flags.DEFINE_string('train_dir', None, 'Where to save decoder checkpoints.')
flags.DEFINE_string('ffn_ckpt', None, 'Load this up as the encoder.')
flags.DEFINE_integer('max_steps', 10000, 'Number of decoder train steps.')


FLAGS = flags.FLAGS


# ------------------------------- main --------------------------------


def main(argv):
    # Parse args a little -----------------------------------------
    fov_size = [FLAGS.fov_len, FLAGS.fov_len, FLAGS.fov_len]

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
    fixed_seed_batch = inputs.fixed_seed_batch(
        FLAGS.batch_size,
        fov_size,
        FLAGS.seed_pad,
        FLAGS.seed_init,
        with_init=(not FLAGS.flat_seed),
    )

    # Load FFN weights ------------------------------------------------
    # Hooking graphs together... This bit loads up weights.
    logging.info('Loading encoder')
    encoder = models.ConvStack3DEncoder.from_ffn_ckpt(
        FLAGS.ffn_ckpt,
        FLAGS.ffn_delta,
        fov_size,
        FLAGS.batch_size,
        fixed_seed_batch,
        depth=FLAGS.layer,
    )

    # Decoder graph ---------------------------------------------------
    decoding_graph = tf.Graph()
    with decoding_graph.as_default():
        # Init encoder ------------------------------------------------
        encoder.define_tf_graph()

        # Build decoder -----------------------------------------------
        logging.info('Init decoder')
        decoder = models.ConvStack3DDecoder(
            fov_size=fov_size,
            batch_size=FLAGS.batch_size,
            loss_lambda=FLAGS.loss_lambda,
            depth=FLAGS.layer,
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
            save_summaries_secs=30,
            save_checkpoint_secs=600,
        ) as sess:
            # Train decoder
            for i, fov_batch in enumerate(fov_batches):
                # Run decoder train op
                sess.run(
                    decoder.train_op,
                    feed_dict={
                        encoder.input_patches: fov_batch,
                        decoder.target: fov_batch,
                    },
                )

                if i > FLAGS.max_steps:
                    print('Reached max_steps', i)
                    break


# ---------------------------------------------------------------------
if __name__ == '__main__':
    flags.mark_flag_as_required('layer')
    flags.mark_flag_as_required('volume_spec')
    flags.mark_flag_as_required('train_dir')
    flags.mark_flag_as_required('ffn_ckpt')
    app.run(main)
