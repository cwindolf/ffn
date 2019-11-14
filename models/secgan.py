import numpy as np
import tensorflow as tf
from training.models import convstacktools
from util.ops import center_crop_vol
from ffn.training.models import convstack_3d
from ffn.training.optimizer import optimizer_from_flags


# ------------------------------ library ------------------------------


def convstack_generator(net, depth=8):
    '''At hand! quoth Pickpurse.

    In the generator, each residual module consists of two 3d
    convolutions with (3, 3, 3) kernels, 32 feature maps, operating in
    VALID mode, with the ReLU activation function used for first
    convolution and linear activation used for the second convolution.
    The residual skip connections used center-cropping of the source to
    match the target volume size. The output of the last residual
    module was passed through a pointwise convolution layer with a
    single featuremap and ​tanh​ activation function to form the
    generated image.
    '''
    conv = tf.contrib.layers.conv3d
    with tf.contrib.framework.arg_scope(
        [conv], num_outputs=32, kernel_size=(3, 3, 3), padding='VALID'
    ):
        # Encoding stack
        net = conv(net, scope='conv0_a')
        net = conv(net, scope='conv0_b', activation_fn=None)

        for i in range(1, depth):
            with tf.name_scope('residual%d' % i):
                # Center crop the residuals
                in_net = net[:, 2:-2, 2:-2, 2:-2, :]

                # Layers
                net = tf.nn.relu(net)
                net = conv(net, scope='conv%d_a' % i)
                net = conv(net, scope='conv%d_b' % i, activation_fn=None)

                # Add residuals
                net += in_net

    net = tf.nn.relu(net)
    logits = conv(
        net,
        num_outputs=1,
        kernel_size=(1, 1, 1),
        activation_fn='tanh',
        scope='conv_lom',
    )

    return logits


def resnet18_discriminator(net):
    '''
    This proverbial saying probably arose from the pick-purse always
    seizing upon the prey nearest him: his maxim being that of the
    Pope's man of gallantry:
        "The thing at hand is of all things the best."

    Enter CHAMBERLAIN:

    The architecture of the discriminator followed that of ResNet-18,
    but we eliminated the finalclassification layer and the global
    average pooling preceding it, and used the output of the last
    residual module directly, which we found necessary for the
    network to train stably.

    Following original lua impl:
    github.com/facebookarchive/fb.resnet.torch/blob/master/models/resnet.lua
    and
    https://github.com/dalgu90/resnet-18-tensorflow/blob/master/resnet.py
    '''
    conv = tf.contrib.layers.conv3d

    # conv1
    with tf.variable_scope('conv1'):
        net = conv(
            net,
            activation_fn='relu',
            scope='conv1',
            num_outputs=64,
            stride=[2, 2, 2],
            kernel_size=[7, 7, 7],
            padding='SAME',
        )

        # max pool 3x3 blocks, stride 2, zero pad 1 px
        net = tf.nn.max_pool3d(
            net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME'
        )

    # basic res layer 64 outputs, 2 repeats, stride 1
    with tf.contrib.framework.arg_scope(
        [conv],
        num_outputs=64,
        kernel_size=(3, 3, 3),
        padding='SAME',
        stride=(1, 1, 1),
    ):
        with tf.variable_scope('conv2_1'):
            shortcut = net
            net = conv(net, activation_fn='relu', scope='conv1')
            net = conv(net, activation_fn=None, scope='conv2')
            net = net + shortcut
            net = tf.nn.relu(net)
        with tf.variable_scope('conv2_2'):
            shortcut = net
            net = conv(net, activation_fn='relu', scope='conv1')
            net = conv(net, activation_fn=None, scope='conv2')
            net = net + shortcut
            net = tf.nn.relu(net)

    # basic res layer 128 outputs, 2 repeats, stride 2
    # basic res layer 256 outputs, 2 repeats, stride 2
    # basic res layer 512 outputs, 2 repeats, stride 2
    for layer, channels in enumerate((12, 256, 512), start=3):
        with tf.contrib.framework.arg_scope(
            [conv], num_outputs=channels, kernel_size=(3, 3, 3), padding='SAME'
        ):
            with tf.variable_scope(f'conv{layer}_1'):
                shortcut = tf.nn.max_pool3d(
                    net, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME'
                )
                net = conv(
                    net, activation_fn='relu', scope='conv1', stride=(2, 2, 2)
                )
                net = conv(
                    net, activation_fn=None, scope='conv2', stride=(1, 1, 1)
                )
                net = net + shortcut
                net = tf.nn.relu(net)
            with tf.variable_scope(f'conv{layer}_2'):
                shortcut = net
                net = conv(
                    net, activation_fn='relu', scope='conv1', stride=(1, 1, 1)
                )
                net = conv(
                    net, activation_fn=None, scope='conv2', stride=(1, 1, 1)
                )
                net = net + shortcut
                net = tf.nn.relu(net)

    with tf.variable_scope('probs'):
        # total spatial average pooling
        net = tf.reduce_mean(net, axis=(1, 2, 3))

        # linear output layer
        probs = tf.contrib.layers.fully_connected(
            net, 1, activation_fn='sigmoid'
        )

    return probs


# ------------------------------ SECGAN -----------------------------


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

    def __init__(
        self,
        batch_size,
        ffn_ckpt,
        generator_conv_clip=16,
        ffn_fov_shape=(33, 33, 33),
        ffn_depth=12,
        cycle_lambda=2.5,
        generator_lambda=1.0,
        generator_seg_lambda=1.0,
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
        '''
        self.ffn_depth = ffn_depth

        self.cycle_lambda = cycle_lambda
        self.generator_lambda = generator_lambda
        self.generator_seg_lambda = generator_seg_lambda

        # Compute input shapes
        # Since the generators use VALID padding, we need to
        # grab more raw data when we're passing through more generators.
        # Placeholders will be sized by the largest input needed from that
        # data, and will be center cropped to suit the task.
        ffn_fov_shape = np.array(ffn_fov_shape)
        self.ffn_input_shape = [batch_size, *ffn_fov_shape, 1]
        self.gen_input_shape = [
            batch_size,
            *(ffn_fov_shape + generator_conv_clip),
            1,
        ]
        self.cycle_input_shape = [
            batch_size,
            *(ffn_fov_shape + 2 * generator_conv_clip),
            1,
        ]

        # Load up FFN -------------------------------------------------
        self.ffn_weights = self.load_ffn_ckpt(ffn_ckpt)

        # Things to be set to placeholders ----------------------------
        # Real inputs
        self.input_labeled = None  # 65
        self.input_unlabeled = None  # 49

        # Fake inputs to fool discriminator
        # TODO: During training, take from a pool of recent output rather
        #       than just the last timestep.
        self.fake_labeled = None  # 33
        self.fake_unlabeled = None  # 33

        # Things which will be output by net --------------------------
        self.generated_labeled = None  # 33
        self.cycle_generated_labeled = None  # 33
        self.generated_unlabeled = None  # 33

    def define_tf_graph(self):
        # Set up placeholders -----------------------------------------
        self.input_labeled = tf.placeholder(
            tf.float32, self.cycle_input_shape, 'input_labeled'
        )
        input_labeled_33 = self.input_labeled[:, 16:-16, 16:-16, 16:-16, :]
        self.input_unlabeled = tf.placeholder(
            tf.float32, self.gen_input_shape, 'input_unlabeled'
        )
        input_unlabeled_33 = self.input_unlabeled[:, 16:-16, 16:-16, 16:-16, :]
        self.fake_labeled = tf.placeholder(
            tf.float32, self.input_shape, 'fake_labeled'
        )
        self.fake_unlabeled = tf.placeholder(
            tf.float32, self.input_shape, 'fake_unlabeled'
        )

        # Generator ops -----------------------------------------------
        # generator G makes fake unlabeled data
        with tf.variable_scope('generator_G') as scope:
            self.generated_unlabeled = convstack_generator(self.input_labeled)
            generated_unlabeled_33 = self.generated_unlabeled[
                :, 8:-8, 8:-8, 8:-8, :
            ]
            G_vars = scope.global_variables()

        # generator F makes fake labeled data
        with tf.variable_scope('generator_F') as scope:
            self.generated_labeled = convstack_generator(self.input_unlabeled)
            scope.reuse_variables()
            self.cycle_generated_labeled = convstack_generator(
                self.generated_unlabeled
            )
            F_vars = scope.global_variables()

        # Segmentation ops --------------------------------------------
        with tf.variable_scope('segmentation') as scope:
            seg_true = convstacktools.fixed_convstack_3d(
                input_labeled_33, self.ffn_weights, depth=self.ffn_depth
            )
            scope.reuse_variables()
            seg_fake = convstacktools.convstack_3d(self.fake_labeled)

        # Discriminator ops -------------------------------------------
        # discriminator for unlabeled data
        with tf.variable_scope('discriminator_D_u') as scope:
            D_u_true = resnet18_discriminator(self.input_unlabeled)
            scope.reuse_variables()
            D_u_fake = resnet18_discriminator(self.fake_unlabeled)
            D_u_vars = scope.global_variables()

        # and for labeled data...
        with tf.variable_scope('discriminator_D_l') as scope:
            D_l_true = resnet18_discriminator(self.input_labeled)
            scope.reuse_variables()
            D_l_fake = resnet18_discriminator(self.fake_labeled)
            D_l_vars = scope.global_variables()

        # and for segmentation...
        with tf.variable_scope('discriminator_D_s') as scope:
            D_S_true = resnet18_discriminator(seg_true - 0.5)
            scope.reuse_variables()
            D_S_fake = resnet18_discriminator(seg_fake - 0.5)
            D_S_vars = scope.global_variables()

        # Loss --------------------------------------------------------
        # Loss for generators
        cycle_consistency = tf.reduce_mean(
            tf.abs(self.cycle_generated_labeled - input_labeled_33)
        )
        G_gan_loss = tf.reduce_mean(tf.square(D_u_fake - 1))
        G_seg_gan_loss = tf.reduce_mean(tf.square(D_S_fake - 1))
        F_gan_loss = tf.reduce_mean(tf.square(D_l_fake - 1))
        G_total_loss = (
            self.cycle_lambda * cycle_consistency
            + self.generator_lambda * G_gan_loss
            + self.generator_seg_lambda * G_seg_gan_loss
        )
        F_total_loss = (
            self.cycle_lambda * cycle_consistency
            + self.generator_lambda * F_gan_loss
        )

        # Loss for discriminators
        D_u_loss = tf.reduce_mean(tf.square(D_u_fake)) + tf.reduce_mean(
            tf.square(D_u_true - 1)
        )
        D_l_loss = tf.reduce_mean(tf.square(D_l_fake)) + tf.reduce_mean(
            tf.square(D_l_true - 1)
        )
        D_S_loss = tf.reduce_mean(tf.square(D_S_fake)) + tf.reduce_mean(
            tf.square(D_S_true - 1)
        )
        D_S_and_l_total_loss = D_l_loss + D_S_loss

        # Optimization ops --------------------------------------------
        # Build optimizers
        G_optimizer = optimizer_from_flags()
        F_optimizer = optimizer_from_flags()
        D_u_optimizer = optimizer_from_flags()
        D_S_and_l_optimizer = optimizer_from_flags()

        # Get some train ops
        # The fact is, everyone needs their own global step.
        # They specify this ordering, why not, let's enforce it.
        G_train_op = G_optimizer.minimize(
            G_total_loss,
            global_step=tf.Variable(0, trainable=False),
            var_list=G_vars,
        )
        with tf.control_dependencies([G_train_op]):
            D_S_and_l_train_op = D_S_and_l_optimizer.minimize(
                D_S_and_l_total_loss,
                global_step=tf.Variable(
                    0, trainable=False, var_list=(D_S_vars + D_l_vars)
                ),
            )
            with tf.control_dependencies([D_S_and_l_train_op]):
                F_train_op = F_optimizer.minimize(
                    F_total_loss,
                    global_step=tf.Variable(0, trainable=False),
                    var_list=F_vars,
                )
                with tf.control_dependencies([F_train_op]):
                    D_u_train_op = D_u_optimizer.minimize(
                        D_u_loss,
                        global_step=tf.Variable(0, trainable=False),
                        var_list=D_u_vars,
                    )
                    with tf.control_dependencies([D_u_train_op]):
                        self.train_op = tf.no_op(name='train_everyone')

        # Add summaries -----------------------------------------------

    @staticmethod
    def load_ffn_ckpt(
        ffn_ckpt, fov_size, batch_size, decoder_ckpt, layer, depth=9
    ):
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
