from absl import flags

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
