import logging
import tensorflow as tf
from absl import app
from absl import flags
from training import inputs
from models.secgan import SECGAN
from util.fakepool import FakePool
import preprocessing.data_util as dx


# ------------------------------- flags -------------------------------

# Training params
flags.DEFINE_string('train_dir', None, 'Where to save decoder checkpoints.')
flags.DEFINE_string('ffn_ckpt', None, 'Load this up as the encoder.')
flags.DEFINE_string('ffn_fov_size', 33, '')
flags.DEFINE_integer('max_steps', 10000, 'Number of decoder train steps.')
flags.DEFINE_integer('batch_size', 8, 'Simultaneous volumes.')

# Data
flags.DEFINE_string(
    'labeled_volume_spec', None, 'Datspec for labeled data volume.'
)
flags.DEFINE_string(
    'unlabeled_volume_spec', None, 'Datspec for unlabeled data volume.'
)

# Model?


FLAGS = flags.FLAGS

# ---------------------------------------------------------------------


def train_secgan(
    labeled_volume_spec,
    unlabeled_volume_spec,
    ffn_ckpt,
    max_steps=10000,
    batch_size=8,
    ffn_fov_size=33,
    generator_clip=16,
):
    '''Run secgan training protocol.'''
    # Load data -------------------------------------------------------
    logging.info('Loading data...')

    labeled_volume = dx.loadspec(labeled_volume_spec)
    unlabeled_volume = dx.loadspec(unlabeled_volume_spec)

    # Make batch generators
    batches_L = inputs.batch_by_fovs(
        labeled_volume, ffn_fov_size + 2 * generator_clip
    )
    batches_U = inputs.batch_by_fovs(
        unlabeled_volume, ffn_fov_size + generator_clip
    )

    # Make seed
    seed = inputs.fixed_seed_batch(batch_size, ffn_fov_size, 0.5, 0.95)

    # Make fake image pool
    pool_L = FakePool()
    pool_U = FakePool()

    # Init model ------------------------------------------------------
    logging.info('Initialize model...')
    secgan = SECGAN(
        batch_size,
        ffn_ckpt,
        generator_conv_clip=generator_clip,
        ffn_fov_shape=(ffn_fov_size, ffn_fov_size, ffn_fov_size),
        ffn_depth=12,
        cycle_lambda=2.5,
        generator_lambda=1.0,
        generator_seg_lambda=1.0,
        input_seed=seed,
    )

    # Enter TF world --------------------------------------------------
    with tf.Graph().as_default():
        # Init model graph
        logging.info('Building graph...')
        secgan.define_tf_graph()

        scaffold = tf.train.Scaffold(
            saver=secgan.saver, summary_op=tf.summary.merge_all()
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
            for i, (batch_L, batch_U) in enumerate(zip(batches_L, batches_U)):
                # get previously generated images
                gen_L, gen_U = sess.run(
                    [secgan.generated_labeled, secgan.generated_unlabeled],
                    feed_dict={
                        secgan.input_labeled: batch_L,
                        secgan.input_unlabeled: batch_U,
                    },
                )

                # query pool
                fake_L = pool_L.query(gen_L)
                fake_U = pool_U.query(gen_U)

                # run train op
                sess.run(
                    [secgan.train_op],
                    feed_dict={
                        secgan.input_labeled: batch_L,
                        secgan.input_unlabeled: batch_U,
                        secgan.fake_labeled: fake_L,
                        secgan.fake_unlabeled: fake_U,
                    },
                )

                if i > max_steps:
                    print('Reached max_steps', i)
                    break


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------------------------------------------
    flags.mark_flags_as_required(
        [
            'train_dir',
            'ffn_ckpt',
            'labeled_volume_spec',
            'unlabeled_volume_spec',
        ]
    )

    def main(argv):
        train_secgan(
            FLAGS.labeled_volume_spec,
            FLAGS.unlabeled_volume_spec,
            FLAGS.ffn_ckpt,
            max_steps=FLAGS.max_steps,
            batch_size=FLAGS.batch_size,
            ffn_fov_size=FLAGS.ffn_fov_size,
        )

    app.run(main)
