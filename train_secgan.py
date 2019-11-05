import logging
import tensorflow as tf
from absl import app
from absl import flags


# ------------------------------- flags -------------------------------

# Training params
flags.DEFINE_string('train_dir', None, 'Where to save decoder checkpoints.')
flags.DEFINE_string('ffn_ckpt', None, 'Load this up as the encoder.')
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

# -------------------------------- main -------------------------------


def main(argv):
    pass


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
    app.run(main)
