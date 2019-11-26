import logging
import numpy as np
import tensorflow as tf
from models import convstacktools
from models import discriminators
from models.convstack_3d_generator import convstack_generator
from util import tfx
from ffn.training.models import convstack_3d
from ffn.training.optimizer import optimizer_from_flags


class SECGAN:
    '''Segmentation-enhanced CycleGAN

    M. Januszewski + V. Jain, 2019:
    https://www.biorxiv.org/content/10.1101/548081v1

    Setting up some notation here for reference when reading
    the code below.

    Generators:
    -----------
        - G : { labeled } -> { unlabeled }
        - F : { unlabeled } -> { labeled }
    The VALID padding means that the output of these is 16 voxels
    smaller than the input on all spatial dims. So, we'll need to
    take in inputs that are larger than the regions we really care
    about, and occasionally center crop to make shapes work out.

    Discriminators:
    ---------------
        - D_l : { labeled } + { labeled-ish } -> [0, 1]
        - D_u : { unlabeled } + { unlabeled-ish } -> [0, 1]
        - D_S : { segmentation } -> [0, 1]
    In this first attempt, we will choose to have the discriminators
    take in input which is the same size as the FFN's input, 33^3. So,
    generators that are generating input for these discriminators will
    need to take in input at least 49^3.

    Segmenter:
    ----------
        - S : {labeled} + { labeled-ish } -> { segmentation }
    '''

    SEEMS_REAL = -0.8
    SEEMS_FAKE = 0.8

    def __init__(
        self,
        batch_size=8,
        ffn_ckpt=None,
        generator_conv_clip=4,
        ffn_fov_shape=(33, 33, 33),
        ffn_depth=12,
        cycle_l_lambda=2.0,
        cycle_u_lambda=0.5,
        generator_lambda=1.0,
        discriminator_lambda=1.0,
        generator_seg_lambda=1.0,
        input_seed=None,
        generator_norm=None,
        discriminator_norm='instance',
        disc_early_maxpool=False,
        discriminator='resnet18',
        convdisc_depth=3,
        generator_depth=8,
        generator_channels=32,
        generator_dropout=False,
        seg_enhanced=True,
        label_noise=0.05,
        inference_ckpt=None,
        inference_input_shape=None,
    ):
        '''
        Arguments
        ---------
        batch_size : int
        ffn_ckpt : string
            Checkpoint of FFN to use for S.
        generator_conv_clip : int
            How many pixels do the generators remove from spatial
            dimension due to their VALID convolutions?
            16 = 8 blocks * 2 convs per block * 2 pixels per conv
        ffn_fov_shape : 3 ints
            Input FOV spatial shape.
        ffn_depth : int
            # of conv-residual blocks in the FFN network
        *_lambda : floats
            Relative weights of different losses.
        input_seed : None or np.array
            Pass in a numpy array to store the seed as a constant.
            Otherwise, feed it into the placeholder self.input_seed.
        '''
        # Log config at init
        paramstrs = [
            f'batch_size={batch_size}',
            f'ffn_ckpt={ffn_ckpt}',
            f'generator_conv_clip={generator_conv_clip}',
            f'ffn_fov_shape={ffn_fov_shape}',
            f'ffn_depth={ffn_depth}',
            f'cycle_l_lambda={cycle_l_lambda}',
            f'cycle_u_lambda={cycle_u_lambda}',
            f'generator_lambda={generator_lambda}',
            f'discriminator_lambda={discriminator_lambda}',
            f'generator_seg_lambda={generator_seg_lambda}',
            f'generator_norm={generator_norm}',
            f'discriminator_norm={discriminator_norm}',
            f'disc_early_maxpool={disc_early_maxpool}',
            f'discriminator={discriminator}',
            f'convdisc_depth={convdisc_depth}',
            f'generator_depth={generator_depth}',
            f'generator_channels={generator_channels}',
            f'generator_dropout={generator_dropout}',
            f'seg_enhanced={seg_enhanced}',
            f'label_noise={label_noise}',
            f'inference_ckpt={inference_ckpt}',
            f'inference_input_shape={inference_input_shape}',
        ]
        paramstr = ",\n    ".join(paramstrs)
        logging.info(f'SECGAN(\n    {paramstr}\n)')

        self.ffn_depth = ffn_depth
        self.generator_conv_clip = generator_conv_clip
        self.label_noise = label_noise

        self.cycle_l_lambda = cycle_l_lambda
        self.cycle_u_lambda = cycle_u_lambda
        self.generator_lambda = generator_lambda
        self.discriminator_lambda = discriminator_lambda
        self.generator_seg_lambda = generator_seg_lambda

        self.gnorm = generator_norm
        self.gdepth = generator_depth
        self.seg_enhanced = seg_enhanced

        self.inference_ckpt = inference_ckpt

        # Make discriminator factory
        if discriminator == 'resnet18':

            def disc(net):
                return discriminators.resnet18(
                    net,
                    norm=discriminator_norm,
                    early_maxpool=disc_early_maxpool,
                )

        elif discriminator == 'convdisc':

            def disc(net):
                return discriminators.convdisc(
                    net, norm=discriminator_norm, depth=convdisc_depth
                )

        else:
            raise ValueError(f'Unknown discriminator {discriminator}')
        self.disc = disc

        # Make generator factory
        def gen(net):
            return convstack_generator(
                net,
                depth=generator_depth,
                dropout=generator_dropout,
                norm=generator_norm,
            )

        self.gen = gen

        # Compute input shapes
        # Since the generators use VALID padding, we need to
        # grab more raw data when we're passing through more generators.
        # Placeholders will be sized by the largest input needed from that
        # data, and will be center cropped to suit the task.
        ffn_fov_shape = np.array(ffn_fov_shape)
        self.ffn_fov_shape = ffn_fov_shape
        self.ffn_input_shape = [batch_size, *ffn_fov_shape, 1]
        self.disc_input_shape = [
            batch_size,
            *(ffn_fov_shape + generator_conv_clip * self.gdepth),
            1,
        ]
        self.cycle_input_shape = [
            batch_size,
            *(ffn_fov_shape + 2 * generator_conv_clip * self.gdepth),
            1,
        ]

        # Load up FFN -------------------------------------------------
        self.ffn_weights = None
        if ffn_ckpt is not None:
            self.ffn_weights = self.load_ffn_ckpt(
                ffn_ckpt, ffn_fov_shape, batch_size
            )
        else:
            assert not seg_enhanced

        # Handle seed
        self.seed_as_placeholder = True
        if input_seed is not None:
            self.seed_as_placeholder = False
            self.input_seed = input_seed

        # Things to be set to placeholders ----------------------------
        # Real inputs
        self.input_labeled = None  # bigg
        self.input_unlabeled = None  # bigg

        # Fake inputs to fool discriminator
        self.fake_labeled = None  # smol
        self.fake_unlabeled = None  # smol

        # Things which will be output by net --------------------------
        self.generated_labeled = None  # smol
        self.cycle_generated_labeled = None  # smol
        self.generated_unlabeled = None  # smol

    def define_tf_graph(self):
        # Set up placeholders -----------------------------------------
        self.input_labeled = tf.placeholder(
            tf.float32, self.cycle_input_shape, 'input_labeled'
        )
        logging.info(f' - Input labeled shape {self.input_labeled.shape}')
        input_labeled_disc = tfx.center_crop_vol(
            self.input_labeled, (self.generator_conv_clip * self.gdepth) // 2
        )
        input_labeled_smol = tfx.center_crop_vol(
            self.input_labeled, self.generator_conv_clip * self.gdepth
        )
        self.input_unlabeled = tf.placeholder(
            tf.float32, self.cycle_input_shape, 'input_unlabeled'
        )
        input_unlabeled_smol = tfx.center_crop_vol(
            self.input_unlabeled, self.generator_conv_clip * self.gdepth
        )
        logging.info(f' - Input unlabeled shape {self.input_unlabeled.shape}')
        input_unlabeled_disc = tfx.center_crop_vol(
            self.input_unlabeled, (self.generator_conv_clip * self.gdepth) // 2
        )
        self.fake_labeled = tf.placeholder(
            tf.float32, self.disc_input_shape, 'fake_labeled'
        )
        fake_labeled_smol = tfx.center_crop_vol(
            self.fake_labeled, (self.generator_conv_clip * self.gdepth) // 2
        )
        self.fake_unlabeled = tf.placeholder(
            tf.float32, self.disc_input_shape, 'fake_unlabeled'
        )

        # Generator ops -----------------------------------------------
        # generator G makes fake unlabeled data
        with tf.variable_scope('generator_G') as scope:
            self.generated_unlabeled = convstack_generator(
                self.input_labeled, norm=self.gnorm, depth=self.gdepth
            )
            gen_unlabeled_smol = tfx.center_crop_vol(
                self.generated_unlabeled,
                (self.generator_conv_clip * self.gdepth) // 2,
            )
            shp = self.generated_unlabeled.shape
            logging.info(f' - Generated unlabeled shape {shp}')
            G_vars = scope.global_variables()

        # generator F makes fake labeled data
        with tf.variable_scope('generator_F') as scope:
            # This is disc shaped
            self.generated_labeled = self.gen(self.input_unlabeled)
            logging.info(
                f' - Generated labeled shape {self.generated_labeled.shape}'
            )
            gen_labeled_smol = tfx.center_crop_vol(
                self.generated_labeled,
                (self.generator_conv_clip * self.gdepth) // 2,
            )
            scope.reuse_variables()
            # This is ffn shaped
            self.cycle_generated_labeled = self.gen(self.generated_unlabeled)
            logging.info(
                ' - Cycle generated labeled shape '
                f'{self.cycle_generated_labeled.shape}'
            )
            F_vars = scope.global_variables()

        # Back into G to close the other cycle
        with tf.variable_scope('generator_G', reuse=True) as scope:
            # This is ffn shaped
            self.cycle_generated_unlabeled = self.gen(self.generated_labeled)

        # Segmentation ops --------------------------------------------
        if self.ffn_weights:
            with tf.variable_scope('segmentation') as scope:
                # Handle seed placeholder logic
                if self.seed_as_placeholder:
                    self.input_seed = tf.placeholder(
                        tf.float32,
                        shape=self.ffn_input_shape,
                        name='input_seed',
                    )
                else:
                    self.input_seed = tf.constant(self.input_seed)

                # Concatenate segmentation inputs with the seed
                seg_true_input = tf.concat(
                    [input_labeled_smol, self.input_seed], axis=4
                )
                seg_fake_input = tf.concat(
                    [fake_labeled_smol, self.input_seed], axis=4
                )
                seg_gen_input = tf.concat(
                    [gen_labeled_smol, self.input_seed], axis=4
                )

                # Actually build FFN
                seg_true = convstacktools.fixed_convstack_3d(
                    seg_true_input, self.ffn_weights, depth=self.ffn_depth
                )
                scope.reuse_variables()
                seg_fake = convstacktools.fixed_convstack_3d(
                    seg_fake_input, self.ffn_weights, depth=self.ffn_depth
                )
                seg_gen = convstacktools.fixed_convstack_3d(
                    seg_gen_input, self.ffn_weights, depth=self.ffn_depth
                )

        # Discriminator ops -------------------------------------------
        # discriminator for unlabeled data
        with tf.variable_scope('discriminator_D_u') as scope:
            D_u_true = self.disc(input_unlabeled_disc)
            scope.reuse_variables()
            D_u_fake = self.disc(self.fake_unlabeled)
            D_u_gen = self.disc(self.generated_unlabeled)
            D_u_vars = scope.global_variables()

        # and for labeled data...
        with tf.variable_scope('discriminator_D_l') as scope:
            D_l_true = self.disc(input_labeled_disc)
            scope.reuse_variables()
            D_l_fake = self.disc(self.fake_labeled)
            D_l_gen = self.disc(self.generated_labeled)
            D_l_vars = scope.global_variables()

        # and for segmentation...
        if self.seg_enhanced:
            with tf.variable_scope('discriminator_D_s') as scope:
                D_S_true = self.disc(seg_true - 0.5)
                scope.reuse_variables()
                D_S_fake = self.disc(seg_fake - 0.5)
                D_S_gen = self.disc(seg_gen - 0.5)
                D_S_vars = scope.global_variables()

        # Loss --------------------------------------------------------
        # Loss for generators
        cycle_consistency_l = tf.reduce_mean(
            tf.abs(self.cycle_generated_labeled - input_labeled_smol)
        )
        cycle_consistency_u = tf.reduce_mean(
            tf.abs(self.cycle_generated_unlabeled - input_unlabeled_smol)
        )
        G_gan_loss = tf.reduce_mean(
            tf.squared_difference(D_u_gen, self.SEEMS_REAL)
        )
        F_gan_loss = tf.reduce_mean(
            tf.squared_difference(D_l_gen, self.SEEMS_REAL)
        )
        G_total_loss = (
            self.cycle_l_lambda * cycle_consistency_l
            + self.cycle_u_lambda * cycle_consistency_u
            + self.generator_lambda * G_gan_loss
        )
        if self.seg_enhanced:
            G_seg_gan_loss = tf.reduce_mean(
                tf.squared_difference(D_S_gen, self.SEEMS_REAL)
            )
            G_total_loss = (
                G_total_loss + self.generator_seg_lambda * G_seg_gan_loss
            )
        F_total_loss = (
            self.cycle_l_lambda * cycle_consistency_l
            + self.cycle_u_lambda * cycle_consistency_u
            + self.generator_lambda * F_gan_loss
        )

        # Loss for discriminators
        D_u_fake_label = tf.random_uniform(
            D_u_fake.shape,
            minval=self.SEEMS_FAKE - self.label_noise,
            maxval=self.SEEMS_FAKE + self.label_noise,
        )
        D_u_true_label = tf.random_uniform(
            D_u_true.shape,
            minval=self.SEEMS_REAL - self.label_noise,
            maxval=self.SEEMS_REAL + self.label_noise,
        )
        D_u_loss = tf.reduce_mean(
            tf.squared_difference(D_u_fake, D_u_fake_label)
        ) + tf.reduce_mean(tf.squared_difference(D_u_true, D_u_true_label))
        D_u_total_loss = self.discriminator_lambda * D_u_loss
        D_l_fake_label = tf.random_uniform(
            D_l_fake.shape,
            minval=self.SEEMS_FAKE - self.label_noise,
            maxval=self.SEEMS_FAKE + self.label_noise,
        )
        D_l_true_label = tf.random_uniform(
            D_l_true.shape,
            minval=self.SEEMS_REAL - self.label_noise,
            maxval=self.SEEMS_REAL + self.label_noise,
        )
        D_l_loss = (
            tf.reduce_mean(tf.squared_difference(D_l_fake, D_l_fake_label))
            + tf.reduce_mean(tf.squared_difference(D_l_true, D_l_true_label))
        )
        D_S_and_l_total_loss = self.discriminator_lambda * D_l_loss
        if self.seg_enhanced:
            D_S_fake_label = tf.random_uniform(
                D_S_fake.shape,
                minval=self.SEEMS_FAKE - self.label_noise,
                maxval=self.SEEMS_FAKE + self.label_noise,
            )
            D_S_true_label = tf.random_uniform(
                D_S_true.shape,
                minval=self.SEEMS_REAL - self.label_noise,
                maxval=self.SEEMS_REAL + self.label_noise,
            )
            D_S_loss = tf.reduce_mean(
                tf.squared_difference(D_S_fake, D_S_fake_label)
            ) + tf.reduce_mean(tf.squared_difference(D_S_true, D_S_true_label))
            D_S_and_l_total_loss = self.discriminator_lambda * (
                D_l_loss + D_S_loss
            )

        # Optimization ops --------------------------------------------
        # Build optimizers
        G_optimizer = optimizer_from_flags()
        F_optimizer = optimizer_from_flags()
        D_u_optimizer = optimizer_from_flags()
        D_S_and_l_optimizer = optimizer_from_flags()

        # Get some train ops
        # The fact is, everyone needs their own global step.
        # They specify this ordering, why not, let's enforce it.
        global_step = tf.train.get_or_create_global_step()
        G_grads_and_vars = G_optimizer.compute_gradients(
            G_total_loss, var_list=G_vars
        )
        G_train_op = G_optimizer.apply_gradients(
            G_grads_and_vars, global_step=global_step
        )
        with tf.control_dependencies([G_train_op]):
            D_S_and_l_vars = D_l_vars
            if self.seg_enhanced:
                D_S_and_l_vars += D_S_vars
            D_S_and_l_grads_and_vars = D_S_and_l_optimizer.compute_gradients(
                D_S_and_l_total_loss, var_list=D_S_and_l_vars
            )
            D_S_and_l_train_op = D_S_and_l_optimizer.apply_gradients(
                D_S_and_l_grads_and_vars
            )
            with tf.control_dependencies([D_S_and_l_train_op]):
                F_grads_and_vars = F_optimizer.compute_gradients(
                    F_total_loss, var_list=F_vars
                )
                F_train_op = F_optimizer.apply_gradients(F_grads_and_vars)
                with tf.control_dependencies([F_train_op]):
                    D_u_grads_and_vars = D_u_optimizer.compute_gradients(
                        D_u_total_loss, var_list=D_u_vars
                    )
                    D_u_train_op = D_u_optimizer.apply_gradients(
                        D_u_grads_and_vars
                    )
                    with tf.control_dependencies([D_u_train_op]):
                        self.train_op = tf.no_op(name='train_everyone')

        # Saving / loading --------------------------------------------
        all_vars = G_vars + F_vars + D_u_vars + D_S_and_l_vars
        self.F_vars = F_vars
        self.saver = tf.train.Saver(var_list=all_vars)

        # Add summaries -----------------------------------------------
        # the various losses
        tf.summary.scalar('losses/cycle_consistency_l', cycle_consistency_l)
        tf.summary.scalar('losses/cycle_consistency_u', cycle_consistency_u)
        tf.summary.scalar('losses/labeled_discriminator_loss', D_l_loss)
        tf.summary.scalar('losses/F_gan_loss', F_gan_loss)
        tf.summary.scalar('losses/unlabeled_discriminator_loss', D_u_loss)
        tf.summary.scalar('losses/G_gan_loss', G_gan_loss)
        tf.summary.scalar(
            'metrics/Du_fake_err',
            tf.reduce_mean(tf.abs(D_u_fake - self.SEEMS_FAKE)),
        )
        tf.summary.scalar(
            'metrics/Du_true_err',
            tf.reduce_mean(tf.abs(D_u_true - self.SEEMS_REAL)),
        )
        tf.summary.scalar(
            'metrics/Dl_fake_err',
            tf.reduce_mean(tf.abs(D_l_fake - self.SEEMS_FAKE)),
        )
        tf.summary.scalar(
            'metrics/Dl_true_err',
            tf.reduce_mean(tf.abs(D_l_true - self.SEEMS_REAL)),
        )
        tf.summary.scalar(
            'metrics/mean_l_true', tf.reduce_mean(input_labeled_smol)
        )
        tf.summary.scalar(
            'metrics/mean_l_fake', tf.reduce_mean(self.generated_labeled)
        )
        tf.summary.scalar(
            'metrics/mean_u_true', tf.reduce_mean(input_unlabeled_smol)
        )
        tf.summary.scalar(
            'metrics/mean_u_fake', tf.reduce_mean(self.generated_unlabeled)
        )
        if self.seg_enhanced:
            tf.summary.scalar(
                'losses/segmentation_discriminator_loss', D_S_loss
            )
            tf.summary.scalar('losses/segmentation_gan_loss', G_seg_gan_loss)
        for grads_and_vars in (
            G_grads_and_vars,
            D_S_and_l_grads_and_vars,
            F_grads_and_vars,
            D_u_grads_and_vars,
        ):
            for grad, var in grads_and_vars:
                try:
                    tf.summary.histogram(
                        'gradients/%s' % var.name.replace(':0', ''), grad
                    )
                except ValueError:
                    logging.info(f'Couldn\'t create grad hist for {var.name}')

        # Images
        # cycle: L -> U' -> L''
        half_fov = self.ffn_fov_shape[0] // 2
        vis_L = input_labeled_smol[:, half_fov, ...]
        vis_U_ = gen_unlabeled_smol[:, half_fov, ...]
        vis_L__ = self.cycle_generated_labeled[:, half_fov, ...]
        vis_cycle_l = tf.concat(
            [vis_L, vis_U_, vis_L__], axis=2, name='vis_cycle_l'
        )
        tf.summary.image('cycles/cycle_labeled', vis_cycle_l)

        # cycle: U -> L' -> U''
        half_fov = self.ffn_fov_shape[0] // 2
        vis_U = input_unlabeled_smol[:, half_fov, ...]
        vis_L_ = gen_labeled_smol[:, half_fov, ...]
        vis_U__ = self.cycle_generated_unlabeled[:, half_fov, ...]
        vis_cycle_u = tf.concat(
            [vis_U, vis_L_, vis_U__], axis=2, name='vis_cycle_u'
        )
        tf.summary.image('cycles/cycle_unlabeled', vis_cycle_u)

        if self.ffn_weights:
            # seg bootstrap: U -> L' -> seg
            vis_seg_gen = tf.concat(
                [vis_U, vis_L_, seg_gen[:, half_fov, ...]],
                axis=2,
                name='vis_seg_gen',
            )
            tf.summary.image('segs/unlabeled_seg', vis_seg_gen)

            # True seg: L -> seg
            vis_seg_true = tf.concat(
                [vis_L, seg_true[:, half_fov, ...]],
                axis=2,
                name='vis_seg_true',
            )
            tf.summary.image('segs/labeled_seg', vis_seg_true)

        # Vis generated against true
        vis_gen_l = tf.concat([vis_L, vis_L_], axis=2, name='vis_gen_l')
        tf.summary.image('gens/labeled', vis_gen_l)
        vis_gen_u = tf.concat([vis_U, vis_U_], axis=2, name='vis_gen_u')
        tf.summary.image('gens/unlabeled', vis_gen_u)

    def define_F_graph(self, inference_input_shape):
        '''Build a minimal graph for running transfer using generator F

        Like, this runs the map from the unlabeled domain to the
        labeled domain.
        '''
        # Placeholder has to be bigg because that's what F is used to.
        self.xfer_input = tf.placeholder(
            tf.float32, (1, *inference_input_shape, 1), 'xfer_input'
        )

        # Build generator F like usual, but with the fixed weights
        with tf.variable_scope('generator_F') as scope:
            generated_labeled_big = self.gen(self.xfer_input)

            F_vars = scope.global_variables()

        # Make an op to restore just the relevant vars
        F_init_op, F_init_fd = tf.contrib.framework.assign_from_checkpoint(
            self.inference_ckpt, F_vars, ignore_missing_vars=True
        )

        # Be sure to run these.
        self.inf_init_op = F_init_op
        self.inf_init_fd = F_init_fd

        # That's it!
        self.xfer_output = generated_labeled_big

    def define_G_graph(self, inference_input_shape):
        '''Build a minimal graph for running transfer using generator G

        Like, this runs the map from the labeled domain to the
        unlabeled domain.
        '''
        # Placeholder has to be bigg because that's what G is used to.
        self.xfer_input = tf.placeholder(
            tf.float32, (1, *inference_input_shape, 1), 'xfer_input'
        )

        # Build generator G like usual, but with the fixed weights
        with tf.variable_scope('generator_G') as scope:
            generated_labeled_big = self.gen(self.xfer_input)

            G_vars = scope.global_variables()

        # Make an op to restore just the relevant vars
        G_init_op, G_init_fd = tf.contrib.framework.assign_from_checkpoint(
            self.inference_ckpt, G_vars, ignore_missing_vars=True
        )

        # Be sure to run these.
        self.inf_init_op = G_init_op
        self.inf_init_fd = G_init_fd

        # That's it!
        self.xfer_output = generated_labeled_big

    @staticmethod
    def load_ffn_ckpt(ffn_ckpt, fov_size, batch_size, depth=12):
        '''Helper to load up FFN weights.
        '''
        logging.info(f' - Loading FFN weights from {ffn_ckpt}')
        # The deltas are not important at all.
        ffn_deltas = [1, 1, 1]
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

        return weights
