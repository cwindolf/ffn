import numpy as np
import tensorflow as tf
import ppx.data_util as dx
from secgan.models import SECGAN


def secgan_infer(unlabeled_volume_spec, output_spec, checkpoint_path):
    '''Run the generator F to map an unabeled volume to a labeled-ish volume
    '''
    # Load data
    unlabeled_volume = dx.loadspec(unlabeled_volume_spec)

    # Process data
    unlabeled_volume = unlabeled_volume.astype(np.float32)
    unlabeled_volume /= 127.5

    # Load model
    generator = SECGAN.genF_from_ckpt(checkpoint_path)

    # TF world
    with tf.Graph().as_default():
        generator.define_inference_graph()

        with tf.Session() as sess:
            xfer_volume = sess.run(
                generator.xfer_output,
                feed_dict={generator.xfer_input: unlabeled_volume},
            )

    # Post process
    # For now, let's just leave this centered and float typed.
    # But multiply by 127.5 since that will be divided out later
    # by the FFN inference script, I think, haha.
    xfer_volume *= 127.5

    # Write output
    dx.writespec(output_spec, xfer_volume)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # -----------------------------------------------------------------

    from absl import app
    from absl import flags

    flags.DEFINE_string('unlabeled_volume_spec', None, '')
    flags.DEFINE_string('output_spec', None, '')
    flags.DEFINE_string('checkpoint_path', None, '')

    FLAGS = flags.FLAGS

    def main(argv):
        secgan_infer(
            FLAGS.unlabeled_volume_spec,
            FLAGS.output_spec,
            FLAGS.checkpoint_path,
        )

    app.run(main)
