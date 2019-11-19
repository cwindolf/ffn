import tensorflow as tf


def convstack_generator(
    net, depth=8, channels=32, dropout=False, norm='instance'
):
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

    # Set up normalization layer
    if norm == 'batch':
        norm = tf.contrib.layers.batch_norm
    elif norm == 'instance':
        norm = tf.contrib.layers.instance_norm
    else:

        def norm(net, scope=None):
            return net

    # Set up dropout layer
    if dropout:
        dropout = tf.contrib.layers.dropout
    else:
        def dropout(net, scope=None):
            return net

    # Build the network.
    with tf.contrib.framework.arg_scope(
        [conv], num_outputs=channels, kernel_size=(3, 3, 3), padding='VALID'
    ):
        # Encoding stack
        net = conv(net, scope='conv0_a', activation_fn=None)
        net = norm(net, scope='norm0_a')
        net = tf.nn.relu(net)
        net = dropout(net, scope='dropout0_a')
        net = conv(net, scope='conv0_b', activation_fn=None)
        net = norm(net, scope='norm0_b')

        for i in range(1, depth):
            with tf.name_scope(f'residual{i}'):
                # Center crop the residuals
                in_net = net[:, 2:-2, 2:-2, 2:-2, :]

                # Layers
                net = tf.nn.relu(net)
                net = dropout(net, scope=f'dropout{i}_a')
                net = conv(net, scope=f'conv{i}_a', activation_fn=None)
                net = norm(net, scope=f'norm{i}_a')
                net = tf.nn.relu(net)
                net = dropout(net, scope=f'dropout{i}_b')
                net = conv(net, scope=f'conv{i}_b', activation_fn=None)
                net = norm(net, scope=f'norm{i}_b')

                # Add residuals
                net += in_net

    net = tf.nn.relu(net)
    logits = conv(
        net,
        num_outputs=1,
        kernel_size=(1, 1, 1),
        activation_fn=tf.tanh,
        scope='gen_output',
    )

    return logits
