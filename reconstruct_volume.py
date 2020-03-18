'''
We want to visualize the grid of reconstructions after the following
transformations:

    - P0, the raw image
    - P1, image -> FFN encoder --> trained decoder
    - P2, image -> trained encoder --> trained decoder

Across layer=2,3,5,7.

This script will reconstruct a volume by encoding with
some encoder and decoding with some decoder. Then you should
go and visualize all of the results.
'''
import logging
from absl import flags
from absl import app

import numpy as np
import tensorflow as tf

from training import inputs
from style_xfer import load_encoder, load_decoder

import ppx.data_util as dx


# -------------------------------- main -------------------------------


def reconstruct_volume(
    inspec,
    outspec,
    layer,
    decoder_ckpt,
    ffn_ckpt=None,
    seed_type='flat',
    encoder_ckpt=None,
    image_mean=128.0,
    image_stddev=33.0,
    fullbyfov=False,
):
    '''Make a grid of reconstructions, and visualize it.
    '''
    # Load data -------------------------------------------------------
    logging.info('Loading data')
    volume = dx.loadspec(inspec)
    volshape = volume.shape

    batch_size = 1
    if fullbyfov:
        batch_size = 128

    fov_size = volume.shape
    if fullbyfov:
        fov_size = [33, 33, 33]

    seed = inputs.fixed_seed_batch(batch_size, fov_size, 0.5, 0.5)
    if seed_type == 'random':
        seed = np.random.uniform(0.05, 0.5, size=seed.shape)

    # Rescale, add batch + feature dim
    volume = (volume.astype(np.float32) - image_mean) / image_stddev
    volume = volume[None, ..., None]

    # Load encoder ----------------------------------------------------
    logging.info('Loading encoder')

    encoder, encoder_graph, einit_fn = load_encoder(
        layer,
        fov_size,
        batch_size=batch_size,
        ffn_ckpt=ffn_ckpt,
        encoder_ckpt=encoder_ckpt,
    )

    # Embed input volumes ----------------------------------------------
    with tf.Session(graph=encoder_graph) as sess:
        einit_fn(sess)

        logging.info('Encoding...')
        if fullbyfov:
            encoding = np.zeros((*volshape, 32))
            counts = np.zeros(encoding.shape, dtype=int)
            batches = inputs.batch_by_fovs(volume, fov_size, batch_size)
            for indices, batch in batches:
                batch_result = sess.run(
                    encoder.encoding,
                    feed_dict={
                        encoder.input_patches: batch,
                        encoder.input_seed: seed,
                    },
                )
                for ind, result in zip(indices, batch_result):
                    encoding[ind] += result
                    counts[ind] += 1

            encoding = encoding / counts
        else:
            encoding = sess.run(
                encoder.encoding,
                feed_dict={
                    encoder.input_patches: volume,
                    encoder.input_seed: seed,
                },
            )

    # Load decoder ----------------------------------------------------
    logging.info(f'Loading decoder from {decoder_ckpt}')

    decoder, decoder_graph, dinit_fn = load_decoder(
        layer, fov_size, decoder_ckpt, batch_size=batch_size
    )

    # Decode transformed features -------------------------------------
    with tf.Session(graph=decoder_graph) as sess:
        dinit_fn(sess)

        logging.info('Decoding transformed features')

        if fullbyfov:
            decoded_transform = np.zeros(volshape)
            counts = np.zeros(decoded_transform.shape, dtype=int)
            batches = inputs.batch_by_fovs(encoding, fov_size, batch_size)
            for indices, batch in batches:
                batch_result = sess.run(
                    decoder.decoding,
                    feed_dict={decoder.input_encoding: batch},
                )
                for ind, result in zip(indices, batch_result):
                    decoded_transform[ind] += result.squeeze()
                    counts[ind] += 1
            decoded_transform = decoded_transform / counts
        else:
            decoded_transform = sess.run(
                decoder.decoding, feed_dict={decoder.input_encoding: encoding}
            )

    # The final unbatching
    decoded_transform = decoded_transform.squeeze()

    # Un-rescale
    decoded_transform *= image_stddev
    decoded_transform += image_mean

    # Write to output volume ------------------------------------------
    logging.info(f'Saving result to {outspec}')
    dx.writespec(outspec, decoded_transform)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Args ------------------------------------------------------------

    # Data
    flags.DEFINE_string('inspec', None, '')
    flags.DEFINE_string('outspec', None, '')
    flags.DEFINE_enum('seed_type', 'flat', ['flat', 'random'], '')

    # Model
    flags.DEFINE_integer('layer', None, '')
    flags.DEFINE_bool('fullbyfov', False, '')
    flags.DEFINE_string('decoder_ckpt', None, '')
    flags.DEFINE_string('encoder_ckpt', None, '')
    flags.DEFINE_string('ffn_ckpt', None, '')

    # OK...
    flags.mark_flag_as_required('inspec')
    flags.mark_flag_as_required('outspec')
    flags.mark_flag_as_required('layer')
    flags.mark_flag_as_required('decoder_ckpt')

    FLAGS = flags.FLAGS

    # Main ------------------------------------------------------------

    def _main(argv):
        reconstruct_volume(
            FLAGS.inspec,
            FLAGS.outspec,
            FLAGS.layer,
            FLAGS.decoder_ckpt,
            ffn_ckpt=FLAGS.ffn_ckpt,
            seed_type=FLAGS.seed_type,
            encoder_ckpt=FLAGS.encoder_ckpt,
            fullbyfov=FLAGS.fullbyfov,
        )

    app.run(_main)
