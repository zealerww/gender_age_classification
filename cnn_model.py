import tensorflow as tf


# based on paper: Age and Gender Classification using Convolutional Neural Networks.
def net_paper(images, training_or_not, out_units):

    use_regular = False
    regular_rate = 0.1
    if use_regular:
        l2_regular = tf.contrib.layers.l2_regularizer(regular_rate)
    else:
        l2_regular = None

    def lkrelu(feature):
        alpha = 0.5
        return tf.maximum(alpha * feature, feature)

    activation = tf.nn.selu
    with tf.variable_scope('block1'):
        x = tf.layers.conv2d(images, 96, [7, 7], [4, 4], padding='VALID', activation=activation, name='conv',
                             kernel_regularizer=l2_regular)
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool')
        x = tf.nn.lrn(x, alpha=0.0001, name='lrn')

    with tf.variable_scope('block2'):
        x = tf.layers.conv2d(x, 256, [5, 5], [1, 1], padding='SAME', activation=activation, name='conv',
                             kernel_regularizer=l2_regular)
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool')
        x = tf.nn.lrn(x, alpha=0.0001, name='lrn')

    with tf.variable_scope('block3'):
        x = tf.layers.conv2d(x, 384, [3, 3], [1, 1], padding='SAME', activation=activation, name='conv',
                             kernel_regularizer=l2_regular)
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool')

    with tf.variable_scope('dense_block'):
        x = tf.layers.flatten(x, name='flatten')
        x = tf.layers.dense(x, 512, activation=activation, name='dense1',
                            kernel_regularizer=l2_regular)
        x = tf.layers.dropout(x, rate=0.5, training=training_or_not, name='dropout1')
        x = tf.layers.dense(x, 512, activation=activation, name='dense2',
                            kernel_regularizer=l2_regular)
        x = tf.layers.dropout(x, rate=0.5, training=training_or_not, name='dropout2')

    y = tf.layers.dense(x, out_units, activation=None, name='y')
    return y


def net_resnet(images, training_or_not, out_units):

    def conv_block(input, filters, strides, train_or_not, name):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(input, filters[0], [1, 1], strides=strides, padding='VALID', name='conv1')
            x = tf.layers.batch_normalization(x, training=training_or_not, name='bn1')
            x = tf.nn.relu(x, name='ac1')

            x = tf.layers.conv2d(x, filters[1], [3, 3], padding='SAME', name='conv2')
            x = tf.layers.batch_normalization(x, training=training_or_not, name='bn2')
            x = tf.nn.relu(x, name='ac2')

            x = tf.layers.conv2d(x, filters[2], [1, 1], padding='VALID', name='conv3')
            x = tf.layers.batch_normalization(x, training=training_or_not, name='bn3')

            short_cut = tf.layers.conv2d(input, filters[2], [1, 1], strides=strides, padding='VALID', name='conv_sc')
            short_cut = tf.layers.batch_normalization(short_cut, training=train_or_not, name='bn_sc')

            out = tf.add(x, short_cut, name='add')
            out = tf.nn.relu(out, name='ac')

        return out

    def identity_block(input, filters, training_or_not, name):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(input, filters[0], [1, 1], name='conv1')
            x = tf.layers.batch_normalization(x, training=training_or_not, name='bn1')
            x = tf.nn.relu(x, name='ac1')

            x = tf.layers.conv2d(x, filters[1], [3, 3], padding='SAME', name='conv2')
            x = tf.layers.batch_normalization(x, training=training_or_not, name='bn2')
            x = tf.nn.relu(x, name='ac2')

            x = tf.layers.conv2d(x, filters[2], [1, 1], name='conv3')
            x = tf.layers.batch_normalization(x, training=training_or_not, name='bn3')

            out = tf.add(x, input, name='add')
            out = tf.nn.relu(out, name='ac')

        return out

    with tf.variable_scope('pre'):
        x = tf.layers.conv2d(images, 32, [3, 3], activation=tf.nn.relu, name='conv1')
        x = tf.layers.max_pooling2d(x, 2, 2, name='pool1')
        x = tf.layers.conv2d(x, 64, [3, 3], activation=tf.nn.relu, name='conv2')
        x = tf.layers.max_pooling2d(x, 2, 2, name='pool2')

    with tf.variable_scope('block1'):
        x = conv_block(x, [64, 64, 256], (2, 2), training_or_not, name='conv_block')
        x = identity_block(x, [64, 64, 256], training_or_not, name='id1')
        x = identity_block(x, [64, 64, 256], training_or_not, name='id2')

    with tf.variable_scope('block2'):
        x = conv_block(x, [128, 128, 512], (2, 2), training_or_not, name='conv_block')
        x = identity_block(x, [128, 128, 512], training_or_not, name='id1')
        x = identity_block(x, [128, 128, 512], training_or_not, name='id2')
        x = identity_block(x, [128, 128, 512], training_or_not, name='id3')

    with tf.variable_scope('block3'):
        x = conv_block(x, [256, 256, 1024], (2, 2), training_or_not, name='conv_block')
        x = identity_block(x, [256, 256, 1024], training_or_not, name='id1')
        x = identity_block(x, [256, 256, 1024], training_or_not, name='id2')
        x = identity_block(x, [256, 256, 1024], training_or_not, name='id3')
        x = identity_block(x, [256, 256, 1024], training_or_not, name='id4')
        x = identity_block(x, [256, 256, 1024], training_or_not, name='id5')

    x = tf.reduce_mean(x, [1, 2], name='average_pool')
    # x = tf.layers.dense(x, 128, activation=tf.nn.relu, name='dense1')
    x = tf.layers.dense(x, out_units, name='dense2')

    return x


model = net_paper





