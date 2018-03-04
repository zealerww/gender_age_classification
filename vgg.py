import tensorflow as tf
import pickle


class VGGFace:
    def __init__(self, weight_path):
        self.weight = None
        with open(weight_path, 'rb') as f:
            self.weight = pickle.load(f)

    def build(self, image, is_training, output_unit):
        with tf.variable_scope('vgg_face'):
            conv1_1 = self.conv_layer(image, 64, 'conv1_1', self.weight[0], self.weight[1])
            conv1_2 = self.conv_layer(conv1_1, 64, 'conv1_2', self.weight[2], self.weight[3])
            pool1 = self.max_pool(conv1_2, 'pool1')

            conv2_1 = self.conv_layer(pool1, 128, 'conv2_1', self.weight[4], self.weight[5])
            conv2_2 = self.conv_layer(conv2_1, 128, 'conv2_2', self.weight[6], self.weight[7])
            pool2 = self.max_pool(conv2_2, 'pool2')

            conv3_1 = self.conv_layer(pool2, 256, 'conv3_1', self.weight[8], self.weight[9])
            conv3_2 = self.conv_layer(conv3_1, 256, 'conv3_2', self.weight[10], self.weight[11])
            conv3_3 = self.conv_layer(conv3_2, 256, 'conv3_3', self.weight[12], self.weight[13])
            pool3 = self.max_pool(conv3_3, 'pool3')

            conv4_1 = self.conv_layer(pool3, 512, 'conv4_1', self.weight[14], self.weight[15])
            conv4_2 = self.conv_layer(conv4_1, 512, 'conv4_2', self.weight[16], self.weight[17])
            conv4_3 = self.conv_layer(conv4_2, 512, 'conv4_3', self.weight[18], self.weight[19])
            pool4 = self.max_pool(conv4_3, 'pool4')

            conv5_1 = self.conv_layer(pool4, 512, 'conv5_1', self.weight[20], self.weight[21])
            conv5_2 = self.conv_layer(conv5_1, 512, 'conv5_2', self.weight[22], self.weight[23])
            conv5_3 = self.conv_layer(conv5_2, 512, 'conv5_3', self.weight[24], self.weight[25])
            pool5 = self.max_pool(conv5_3, 'pool5')

            flatten = tf.layers.flatten(pool5, name='flatten')
        with tf.variable_scope('fine_tune'):
            y = tf.layers.dense(flatten, 512, activation=tf.nn.relu, name='dense1')
            y = tf.layers.dropout(y, 0.5, training=is_training, name='dp1')
            y = tf.layers.dense(y, 512, activation=tf.nn.relu, name='dens2')
            y = tf.layers.dropout(y, 0.5, training=is_training, name='dp2')
            y = tf.layers.dense(y, output_unit, name='y')

        # result = tf.identity(y, name='y_ouput')
        softmax_res = tf.nn.softmax(y, name='softmax')

        # free data
        self.weight = None

        return y

    def conv_layer(self, input, filters, name, kernel_weight, bias_weight, ksize=3):
        with tf.variable_scope(name):
            return tf.layers.conv2d(inputs=input, filters=filters, kernel_size=[ksize, ksize], padding='same',
                                    activation=tf.nn.relu, kernel_initializer=self.get_initial(kernel_weight),
                                    bias_initializer=self.get_initial(bias_weight),
                                    trainable=False)

    def fc_layer(self, input, name, unit, kernel_weight, bias_weight, use_dropout=False, trainable=False, use_relu=True):
        with tf.variable_scope(name):
            if use_relu:
                ac = tf.nn.relu
            else:
                ac = None

            fc = tf.layers.dense(inputs=input, units=unit, activation=ac,
                                 kernel_initializer=self.get_initial(kernel_weight),
                                 bias_initializer=self.get_initial(bias_weight), trainable=trainable)
            dropout = tf.layers.dropout(fc, 0.5, training=use_dropout)
            return dropout

    @staticmethod
    def max_pool(input, name=None, stride=2):
        return tf.layers.max_pooling2d(inputs=input, pool_size=[2, 2], strides=stride, padding='same', name=name)

    @staticmethod
    def get_initial(data):
        return tf.constant_initializer(data)



