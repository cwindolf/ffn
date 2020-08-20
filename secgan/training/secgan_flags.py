from absl import flags

# Training params
flags.DEFINE_string("train_dir", None, "Where to save SECGAN checkpoints.")
flags.DEFINE_string(
    "ffn_ckpt", None, "Path to ffn, only used if --seg_enhanced is passed."
)
flags.DEFINE_integer("ffn_fov_size", 33, "")
flags.DEFINE_integer("ffn_features_layer", 12, "Use 12 for last layer.")
flags.DEFINE_integer("max_steps", 10000, "Number of decoder train steps.")
flags.DEFINE_integer("batch_size", 8, "Simultaneous volumes.")
flags.DEFINE_integer(
    "fakepool_sz",
    0,
    "Size of the 'Fake Pool'. Larger will slow discriminator learning.",
)

# Data
flags.DEFINE_spaceseplist(
    "labeled_volume_specs", None, "Datspecs for labeled data volumes."
)
flags.DEFINE_spaceseplist(
    "unlabeled_volume_specs", None, "Datspecs for unlabeled data volumes."
)

# Model?
flags.DEFINE_string(
    "discriminator",
    "resnet18",
    "What model to use for discriminators. So far, we have resnet18 "
    "and convdisc. convdisc is additionally configured by "
    "--convdisc_depth.",
)
flags.DEFINE_float(
    "cycle_l_lambda",
    2.5,
    "Weight of the cycle consistency loss for the cycle "
    "Labeled -> Unlabeled -> Labeled.",
)
flags.DEFINE_float(
    "cycle_u_lambda",
    0.5,
    "Weight of the cycle consistency loss for the cycle "
    "Unlabeled -> Labeled -> Unlabeled.",
)
flags.DEFINE_float("generator_lambda", 1.0, "Weight of the GAN losses.")
flags.DEFINE_float(
    "generator_seg_lambda",
    1.0,
    "Weight of the GAN loss applied via FFN to the generator mapping U -> L.",
)
flags.DEFINE_float(
    "u_discriminator_lambda",
    1.0,
    "Weight of the loss for the discriminator of unlabeled image data.",
)
flags.DEFINE_float(
    "l_discriminator_lambda",
    1.0,
    "Weight of the loss for the discriminator of labeled data. "
    "Also inherited by discriminator on segmentation.",
)
flags.DEFINE_string(
    "generator_norm",
    None,
    "Normalization for generators. Choose batch, instance, or leave blank. "
    "I don't usually use it.",
)
flags.DEFINE_string(
    "discriminator_norm",
    "instance",
    "Normalization for discriminators. "
    "Choose batch, instance, or leave blank.",
)
flags.DEFINE_boolean(
    "disc_early_maxpool",
    False,
    "Use max pooling early on in discriminators to reduce their size. "
    "Probably not a good idea.",
)
flags.DEFINE_boolean(
    "seg_enhanced",
    True,
    "If supplied, use SECGAN rather than CycleGAN. In that case, "
    "please pass an ffn checkpoint.",
)
flags.DEFINE_boolean("generator_dropout", False, "Use dropout in generators.")
flags.DEFINE_integer(
    "convdisc_depth",
    3,
    "Depth of vanilla CNN discriminator if --discriminator=convdisc.",
)
flags.DEFINE_integer(
    "generator_depth", 8, "Number of residual blocks in generators."
)
flags.DEFINE_integer(
    "generator_channels", 32, "Number of channels in each layer of generators."
)
flags.DEFINE_float(
    "label_noise",
    0.0,
    "Apply noise to labels fed to discriminators to slow their learning.",
)
flags.DEFINE_boolean(
    "seed_logit",
    True,
    "Evidence is fairly conclusive that one should not change this.",
)
