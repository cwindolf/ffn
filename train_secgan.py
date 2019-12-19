import os.path
import logging
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from training import inputs
from models.secgan import SECGAN
from util.fakepool import FakePool


# ------------------------------- flags -------------------------------

# Training params
flags.DEFINE_string('train_dir', None, 'Where to save decoder checkpoints.')
flags.DEFINE_string('ffn_ckpt', None, 'Load this up as the encoder.')
flags.DEFINE_integer('ffn_fov_size', 33, '')
flags.DEFINE_integer('ffn_features_layer', 12, '')
flags.DEFINE_integer('max_steps', 10000, 'Number of decoder train steps.')
flags.DEFINE_integer('batch_size', 8, 'Simultaneous volumes.')
flags.DEFINE_integer('fakepool_sz', 0, '')
flags.DEFINE_boolean('split_devices', False, '')

# Data
flags.DEFINE_spaceseplist(
    'labeled_volume_specs', None, 'Datspecs for labeled data volumes.'
)
flags.DEFINE_spaceseplist(
    'unlabeled_volume_specs', None, 'Datspecs for unlabeled data volumes.'
)

# Model?
flags.DEFINE_string('discriminator', 'resnet18', '')
flags.DEFINE_float('cycle_l_lambda', 2.5, '')
flags.DEFINE_float('cycle_u_lambda', 0.5, '')
flags.DEFINE_float('generator_lambda', 1.0, '')
flags.DEFINE_float('generator_seg_lambda', 1.0, '')
flags.DEFINE_float('u_discriminator_lambda', 1.0, '')
flags.DEFINE_float('l_discriminator_lambda', 1.0, '')
flags.DEFINE_string('generator_norm', None, '')
flags.DEFINE_string('discriminator_norm', 'instance', '')
flags.DEFINE_boolean('disc_early_maxpool', False, '')
flags.DEFINE_boolean('seg_enhanced', True, '')
flags.DEFINE_boolean('generator_dropout', False, '')
flags.DEFINE_integer('convdisc_depth', 3, '')
flags.DEFINE_integer('generator_depth', 8, '')
flags.DEFINE_integer('generator_channels', 32, '')
flags.DEFINE_float('label_noise', 0.0, '')
flags.DEFINE_boolean(
    'seed_logit',
    True,
    'Evidence is fairly conclusive that one should not change this.',
)


FLAGS = flags.FLAGS

# ---------------------------------------------------------------------


def train_secgan(
    labeled_volume_specs,
    unlabeled_volume_specs,
    ffn_ckpt=None,
    max_steps=10000,
    batch_size=8,
    generator_clip=4,
    ffn_fov_size=33,
    ffn_features_layer=12,
    cycle_l_lambda=2.0,
    cycle_u_lambda=0.5,
    generator_lambda=1.0,
    u_discriminator_lambda=1.0,
    l_discriminator_lambda=1.0,
    generator_seg_lambda=1.0,
    generator_norm=None,
    discriminator_norm='instance',
    disc_early_maxpool=False,
    discriminator='resnet18',
    convdisc_depth=3,
    generator_depth=8,
    generator_channels=32,
    generator_dropout=False,
    seg_enhanced=True,
    label_noise=0.0,
    split_devices=False,
    seed_logit=True,
):
    '''Run secgan training protocol.'''
    # Load data -------------------------------------------------------
    logging.info('Loading data...')

    # Make batch generators
    if len(labeled_volume_specs) == 1:
        batches_L = inputs.random_fovs(
            labeled_volume_specs[0],
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
            image_mean=None,
            image_stddev=None,
        )
    else:
        batches_L = inputs.multi_random_fovs(
            labeled_volume_specs,
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
        )
    if len(unlabeled_volume_specs) == 1:
        batches_U = inputs.random_fovs(
            unlabeled_volume_specs[0],
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
            image_mean=None,
            image_stddev=None,
        )
    else:
        batches_U = inputs.multi_random_fovs(
            unlabeled_volume_specs,
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
        )

    # Make seed
    seed = inputs.fixed_seed_batch(
        batch_size,
        ffn_fov_size,
        0.5,
        0.95,
        with_init=True,
        seed_logit=seed_logit,
    )

    # Make fake image pool
    pool_L = FakePool()
    pool_U = FakePool()

    # Init model ------------------------------------------------------
    logging.info('Initialize model...')
    secgan = SECGAN(
        batch_size=batch_size,
        ffn_ckpt=ffn_ckpt,
        generator_conv_clip=generator_clip,
        ffn_fov_shape=(ffn_fov_size, ffn_fov_size, ffn_fov_size),
        ffn_features_layer=ffn_features_layer,
        cycle_l_lambda=cycle_l_lambda,
        cycle_u_lambda=cycle_u_lambda,
        generator_lambda=generator_lambda,
        u_discriminator_lambda=u_discriminator_lambda,
        l_discriminator_lambda=l_discriminator_lambda,
        generator_seg_lambda=generator_seg_lambda,
        input_seed=seed,
        generator_norm=generator_norm,
        discriminator_norm=discriminator_norm,
        disc_early_maxpool=disc_early_maxpool,
        discriminator=discriminator,
        convdisc_depth=convdisc_depth,
        generator_depth=generator_depth,
        generator_channels=generator_channels,
        generator_dropout=generator_dropout,
        seg_enhanced=seg_enhanced,
        label_noise=label_noise,
        F_device=0 if split_devices else None,
        G_device=1 if split_devices else None,
        S_device=0 if split_devices else None,
    )

    # Enter TF world --------------------------------------------------
    with tf.Graph().as_default():
        # Init model graph
        logging.info('Building graph...')
        secgan.define_tf_graph()

        # Training machinery
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
            # Get some initial images to start off the pool.
            batch_L_0 = next(batches_L)
            batch_U_0 = next(batches_U)
            fake_0 = np.zeros(secgan.disc_input_shape, dtype=np.float32)
            gen_L, gen_U = sess.run(
                [secgan.generated_labeled, secgan.generated_unlabeled],
                feed_dict={
                    secgan.input_labeled: batch_L_0,
                    secgan.input_unlabeled: batch_U_0,
                    # Need to be supplied but won't do anything since
                    # we are not running the train op.
                    secgan.fake_labeled: fake_0,
                    secgan.fake_unlabeled: fake_0,
                },
            )

            # Now we can iterate happily.
            for i, (batch_L, batch_U) in enumerate(zip(batches_L, batches_U)):
                # run train op
                _, gen_L, gen_U = sess.run(
                    [
                        secgan.train_op,
                        secgan.generated_labeled,
                        secgan.generated_unlabeled,
                    ],
                    feed_dict={
                        secgan.input_labeled: batch_L,
                        secgan.input_unlabeled: batch_U,
                        secgan.fake_labeled: pool_L.query(gen_L),
                        secgan.fake_unlabeled: pool_U.query(gen_U),
                    },
                )

                if i > max_steps:
                    print('Reached max_steps', i)
                    break


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------------------------------------------
    flags.mark_flags_as_required(
        ['train_dir', 'labeled_volume_specs', 'unlabeled_volume_specs']
    )

    def main(argv):
        flags_str = FLAGS.flags_into_string()
        logging.info(f'train_secgan.py with flags:\n{flags_str}')
        with open(
            os.path.join(FLAGS.train_dir, 'flagfile.txt'), 'w'
        ) as flagfile:
            flagfile.write(flags_str)
        train_secgan(
            FLAGS.labeled_volume_specs,
            FLAGS.unlabeled_volume_specs,
            ffn_ckpt=FLAGS.ffn_ckpt,
            max_steps=FLAGS.max_steps,
            batch_size=FLAGS.batch_size,
            ffn_fov_size=FLAGS.ffn_fov_size,
            ffn_features_layer=FLAGS.ffn_features_layer,
            cycle_l_lambda=FLAGS.cycle_l_lambda,
            cycle_u_lambda=FLAGS.cycle_u_lambda,
            generator_lambda=FLAGS.generator_lambda,
            u_discriminator_lambda=FLAGS.u_discriminator_lambda,
            l_discriminator_lambda=FLAGS.l_discriminator_lambda,
            generator_seg_lambda=FLAGS.generator_seg_lambda,
            generator_norm=FLAGS.generator_norm,
            discriminator_norm=FLAGS.discriminator_norm,
            disc_early_maxpool=FLAGS.disc_early_maxpool,
            discriminator=FLAGS.discriminator,
            convdisc_depth=FLAGS.convdisc_depth,
            generator_depth=FLAGS.generator_depth,
            generator_channels=FLAGS.generator_channels,
            generator_dropout=FLAGS.generator_dropout,
            seg_enhanced=FLAGS.seg_enhanced,
            label_noise=FLAGS.label_noise,
            split_devices=FLAGS.split_devices,
            seed_logit=FLAGS.seed_logit,
        )

    app.run(main)
