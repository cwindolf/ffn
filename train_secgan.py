import os
import time
import logging
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from secgan.training import inputs
from secgan.models import SECGAN
from secgan.util.fakepool import FakePool


# ---------------------------------------------------------------------
# Flags

from training import secgan_flags  # noqa: W0611

# Data parallel training options.
# See also some of the training infra options above.
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
# flags.DEFINE_string('master', '', 'Network address of the master.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.')
flags.DEFINE_list(
    'ps_hosts',
    '',
    'Parameter servers. Comma-separated list of ' '<hostname>:<port> pairs.',
)
flags.DEFINE_list(
    'worker_hosts',
    '',
    'Worker servers. Comma-separated list of ' '<hostname>:<port> pairs.',
)
flags.DEFINE_enum(
    'job_name',
    'worker',
    ['ps', 'worker'],
    'Am I a parameter server or a worker?',
)
flags.DEFINE_integer('replica_step_delay', 100, '')

FLAGS = flags.FLAGS


# ---------------------------------------------------------------------
# Distributed support code

# for done queue reference, see:
# - https://github.com/tensorflow/tensorflow/issues/4713
# - https://gist.github.com/yaroslavvb/ea1b1bae0a75c4aae593df7eca72d9ca


def create_done_queue(i):
    with tf.device("/job:ps/task:%d" % i):
        return tf.FIFOQueue(
            len(FLAGS.worker_hosts),
            tf.int32,
            shared_name="done_queue" + str(i),
        )


def create_done_queues():
    return [create_done_queue(i) for i in range(FLAGS.ps_tasks)]


def get_cluster_spec():
    '''
    Convert the `{--ps_hosts,--worker_hosts}` flags into a definition of the
    distributed cluster we're running on.

    Returns:
    None                  if these flags aren't present, which signals local
                          training.
    tf.train.ClusterSpec  describing the cluster otherwise
    '''
    logging.info(
        '%s %d Building cluster from ps_hosts %s, worker_hosts %s'
        % (FLAGS.job_name, FLAGS.task, FLAGS.ps_hosts, FLAGS.worker_hosts)
    )
    if not (FLAGS.ps_hosts or FLAGS.worker_hosts or FLAGS.ps_tasks):
        return None
    elif FLAGS.ps_hosts and FLAGS.worker_hosts and FLAGS.ps_tasks > 0:
        cluster_spec = tf.train.ClusterSpec(
            {'ps': FLAGS.ps_hosts, 'worker': FLAGS.worker_hosts}
        )
        return cluster_spec
    else:
        raise ValueError(
            'Set either all or none of --ps_hosts, --worker_hosts, --ps_tasks'
        )


# ---------------------------------------------------------------------
# Parameter server


def secgan_parameter_server(cluster_spec=None):
    # Distributed or local training server
    if cluster_spec:
        server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task
        )
    else:
        server = tf.train.Server.create_local_server()

    # Parameter servers wait for instructions
    logging.info('Starting parameter server.')
    queue = create_done_queue(FLAGS.task)
    # Lots of times ps don't need a session -- this one does to
    # listen to its done queue
    sess = tf.Session(
        server.target,
        config=tf.ConfigProto(
            log_device_placement=False, allow_soft_placement=True
        ),
    )

    # Wait for quit queue signals from workers
    for i in range(len(FLAGS.worker_hosts)):
        sess.run(queue.dequeue())
        logging.info('PS' + str(FLAGS.task) + ' got quit sig number ' + str(i))

    # Quit
    logging.info('PS' + str(FLAGS.task) + ' exiting.')
    time.sleep(1.0)
    time.sleep(1.0)
    # For some reason the sess.close is hanging, this hard kill
    # is the only way I can find to exit.
    os._exit(0)


# ---------------------------------------------------------------------
# The actual training routine.


def secgan_training_worker(
    labeled_volume_specs,
    unlabeled_volume_specs,
    # Model flags
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
    seed_logit=True,
    random_state=np.random,
    # Distributed flags
    cluster_spec=None,
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
            random_state=random_state,
        )
    else:
        batches_L = inputs.multi_random_fovs(
            labeled_volume_specs,
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
            random_state=random_state,
        )
    if len(unlabeled_volume_specs) == 1:
        batches_U = inputs.random_fovs(
            unlabeled_volume_specs[0],
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
            image_mean=None,
            image_stddev=None,
            random_state=random_state,
        )
    else:
        batches_U = inputs.multi_random_fovs(
            unlabeled_volume_specs,
            batch_size,
            ffn_fov_size + 2 * generator_clip * generator_depth,
            random_state=random_state,
        )
    batches_LU = zip(batches_L, batches_U)

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
    )

    # Enter TF world --------------------------------------------------

    # Some infrastructure
    is_chief = FLAGS.task == 0
    if cluster_spec:
        server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task
        )
    else:
        server = tf.train.Server.create_local_server()

    with tf.Graph().as_default():
        # If distributed, make ops to coordinate shutdown
        if cluster_spec:
            done_ops = []
            with tf.device('/job:worker/task:%d' % FLAGS.task):
                done_msg = f'Worker {FLAGS.task} signaling parameter servers'
                done_ops = [q.enqueue(1) for q in create_done_queues()] + [
                    tf.Print(tf.constant(0), [], message=done_msg)
                ]

        # Handle lots of devices
        with tf.device(
            tf.train.replica_device_setter(
                ps_tasks=FLAGS.ps_tasks,
                cluster=cluster_spec,
                merge_devices=True,
            )
        ):
            # In some situations, we will add hooks.
            hooks = []

            if cluster_spec:
                # If distributed, make sure the done queue ops run.
                hooks.append(tf.train.FinalOpsHook(done_ops))

            # Distributed communication
            device_filters = None
            if cluster_spec:
                device_filters = [
                    '/job:ps',
                    '/job:worker/task:%d' % FLAGS.task,
                ]

            # Init model graph
            logging.info('Building graph...')
            secgan.define_tf_graph()

            # We'll need the feed dict all over the place.
            # What a pain.
            batch_L_0, batch_U_0 = next(batches_LU)
            feed_dict = {
                secgan.input_labeled: batch_L_0,
                secgan.input_unlabeled: batch_U_0,
                # Need to be supplied but won't do anything since
                # we are not running the train op in the first
                # run call.
                secgan.fake_labeled: np.empty(
                    secgan.disc_input_shape, dtype=np.float32
                ),
                secgan.fake_unlabeled: np.empty(
                    secgan.disc_input_shape, dtype=np.float32
                ),
            }

            # Support for synchronous optimizers
            if FLAGS.synchronous:
                fd = [feed_dict]
                print(f'Running with a synchronous optimizer')
                hooks += [tf.train.FeedFnHook(lambda: fd[0])]
                hooks += (
                    opt.make_session_run_hook(is_chief, num_tokens=0)
                    for opt in secgan.optimizers
                )

            # Training machinery
            scaffold = tf.train.Scaffold(
                saver=secgan.saver,
                summary_op=tf.Print(
                    tf.summary.merge_all(),
                    [secgan.global_step],
                    message='Saving summaries at step ',
                ),
            )
            config = tf.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                device_filters=device_filters,
            )
            with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                config=config,
                scaffold=scaffold,
                checkpoint_dir=FLAGS.train_dir,
                save_summaries_secs=30,
                save_checkpoint_secs=300,
                hooks=hooks,
            ) as sess:
                # If we're not the chief, wait around a bit.
                if not is_chief:
                    while not FLAGS.synchronous:
                        time.sleep(5.0)
                        step = int(sess.run(secgan.global_step))
                        if step > FLAGS.replica_step_delay * FLAGS.task:
                            break
                    logging.info(f'Worker task {FLAGS.task} coming online')

                # Populate generated images before running train op
                # (since the train op depends on the gens.)
                # Need to run this in raw session so that it doesn't
                # run the train op. (the mon session wants to run
                # summaries, which depend on the train op...)
                logging.info('First run without train op')
                gen_L, gen_U = sess._tf_sess().run(
                    [secgan.generated_labeled, secgan.generated_unlabeled],
                    feed_dict=feed_dict,
                )
                assert np.isfinite(gen_L).all() and np.isfinite(gen_U).all()
                logging.info(
                    f'gen_L init stats: '
                    f'{gen_L.min()}, {gen_L.mean()}, {gen_L.max()}'
                )
                logging.info(
                    f'gen_U init stats: '
                    f'{gen_U.min()}, {gen_U.mean()}, {gen_U.max()}'
                )

                # Now we can iterate happily.
                for i, (batch_L, batch_U) in enumerate(batches_LU):
                    # Update feeds
                    feed_dict = {
                        secgan.input_labeled: batch_L,
                        secgan.input_unlabeled: batch_U,
                        secgan.fake_labeled: pool_L.query(gen_L),
                        secgan.fake_unlabeled: pool_U.query(gen_U),
                    }
                    if FLAGS.synchronous:
                        fd[0] = feed_dict

                    # run train op, get new gens.
                    _, gen_L, gen_U = sess.run(
                        [
                            secgan.train_op,
                            secgan.generated_labeled,
                            secgan.generated_unlabeled,
                        ],
                        feed_dict=feed_dict,
                    )

                    if i > max_steps:
                        print('Reached max_steps', i)
                        break

                    if sess.should_stop():
                        print('Session said stop at step', i)
                        break


# ---------------------------------------------------------------------
# Main


def main(argv):
    is_chief = FLAGS.job_name == 'worker' and FLAGS.task == 0

    # Log info about the run
    if is_chief:
        flags_str = FLAGS.flags_into_string()
        logging.info(f'train_secgan.py with flags:\n{flags_str}')
        with open(
            os.path.join(FLAGS.train_dir, 'flagfile.txt'), 'w'
        ) as flagfile:
            flagfile.write(flags_str)

    # Seed np random so batches aren't the same
    seed = int((time.time() * 1000 + FLAGS.task * 3600 * 24) % (2 ** 32))
    logging.info('Random seed: %r', seed)
    random_state = np.random.RandomState(seed)

    # Figure out the world
    cluster_spec = get_cluster_spec()

    if FLAGS.job_name == 'ps':
        secgan_parameter_server(cluster_spec)
    elif FLAGS.job_name == 'worker':
        secgan_training_worker(
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
            seed_logit=FLAGS.seed_logit,
            random_state=random_state,
            cluster_spec=cluster_spec,
        )
    else:
        assert False


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------------------------------------------
    flags.mark_flags_as_required(
        ['train_dir', 'labeled_volume_specs', 'unlabeled_volume_specs']
    )

    app.run(main)
