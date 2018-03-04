import tensorflow as tf
import numpy as np
import os
import time
import cv2
import shutil
import datetime
import vgg
import cnn_model

tf.app.flags.DEFINE_integer('class_num', 2, "Num of classes")
tf.app.flags.DEFINE_integer('image_size', 227, "size of image.")
tf.app.flags.DEFINE_integer('input_size', 192, "size of input image for model.")
tf.app.flags.DEFINE_integer('image_channel', 3, "channel of image.")
tf.app.flags.DEFINE_integer('image_num', 18000, "channel of image.")
tf.app.flags.DEFINE_integer('batch_size', 64, "Num of each batch.")
tf.app.flags.DEFINE_integer('val_batch_size', 64, "Num of each validation batch.")
tf.app.flags.DEFINE_integer('epoch_num', 100, "Num of epochs.")
tf.app.flags.DEFINE_string('train_data_dir', './gender_data/train/', "Train data dir while images in")
tf.app.flags.DEFINE_string('val_data_dir', './gender_data/validation/', "Validation data dir while images in")
tf.app.flags.DEFINE_string('model_path', './gender_model/', "Path the model in")
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', "Model name")
tf.app.flags.DEFINE_bool('restore', False, "If restore model from file")
tf.app.flags.DEFINE_string('test_pic', './test.bmp', "Test picture.")
tf.app.flags.DEFINE_string('char_dict', './char_dict.bin', "Char dict.")
tf.app.flags.DEFINE_string('log_dir', './log_dir', "Tf summary log.")
tf.app.flags.DEFINE_integer('run_type', 0, "0 for traning;1 for testing all model;"
                                           "2 for testing single model;3 for inference")
tf.app.flags.DEFINE_bool('multi_crop', False, "If using multi crop when testing")

FLAGS = tf.app.flags.FLAGS

VGG_FACE_MODEL_WEIGHT = './face.weight'


# reading tf record data
class DataReader:
    def __init__(self, data_dir, bath_size, num_epochs, is_training):
        files = os.listdir(data_dir)
        files = map(lambda x: os.path.join(data_dir, x), files)
        self.file_names = list(files)
        self.batch_size = bath_size
        self.num_epochs = num_epochs
        self.is_training = is_training

    def input(self):
        file_queue = tf.train.string_input_producer(self.file_names, num_epochs=self.num_epochs)
        image, label = self.read_and_decode(file_queue)

        if self.is_training:
            image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.batch_size,
                                                              capacity=5000, min_after_dequeue=1000)
        else:
            image_batch, label_batch = tf.train.batch([image, label], batch_size=self.batch_size, capacity=5000)
        return image_batch, tf.reshape(label_batch, [self.batch_size])

    def read_and_decode(self, file_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # label
        label = tf.cast(features['label'], tf.int32)

        # image data
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel])

        image = self.preprocess(image)

        return image, label

    def preprocess(self, image):
        """
        data augmentation: crop, flip, bright, contrast, scale, roate.
        :param image:
        :return:
        """
        def rotate_image(image):
            angle = tf.random_uniform([], minval=-10*2*np.pi/360, maxval=10*2*np.pi/360)
            image = tf.contrib.image.rotate(image, angle,)
            return image

        def scale_image(image):
            scale_x = tf.random_uniform([], minval=0.8, maxval=1.2)
            scale_y = tf.random_uniform([], minval=0.8, maxval=1.2)
            size_x = tf.cast(tf.multiply(scale_x, tf.cast(image.shape[0], dtype='float32')), dtype='int32')
            size_y = tf.cast(tf.multiply(scale_y, tf.cast(image.shape[1], dtype='float32')), dtype='int32')
            image = tf.image.resize_images(image, [size_x, size_y])
            image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_size, FLAGS.image_size)
            return image

        image = tf.cast(image, 'float32')

        if self.is_training:
            image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, FLAGS.image_channel])
            # image = scale_image(image)
            # image = rotate_image(image)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, 0.5, 1.5)

        image = image * (1. / 255)

        return image


def train():
    # train data batch
    train_data_reader = DataReader(FLAGS.train_data_dir, FLAGS.batch_size, FLAGS.epoch_num, True)
    train_images, train_labels = train_data_reader.input()

    # validation data batch
    val_data_reader = DataReader(FLAGS.val_data_dir, FLAGS.val_batch_size, None, False)
    val_images, val_labels = val_data_reader.input()

    # place holder
    image_batch = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, FLAGS.image_channel],
                                 name='image_batch')
    label_batch = tf.placeholder(dtype=tf.int32, name='label_batch')
    training_or_not = tf.placeholder(dtype=tf.bool, name='training_or_not')

    # build model
    vggface = vgg.VGGFace(VGG_FACE_MODEL_WEIGHT)
    y_result = vggface.build(image_batch, training_or_not, FLAGS.class_num)

    # loss and accuracy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=y_result),
                          name='loss')
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss += reg_loss
    labels_l = tf.cast(label_batch, tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_result, 1), labels_l), tf.float32), name='accuracy')

    # train method
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_method = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

    # train summary op
    train_summary_op = tf.summary.merge([tf.summary.scalar('loss', loss),
                                         tf.summary.scalar('accuracy', accuracy)])

    # validation summary op
    val_loss_pl = tf.placeholder(tf.float32, name='val_loss_pl')
    val_accuracy_pl = tf.placeholder(tf.float32, name='val_accuracy_pl')
    val_summary_op = tf.summary.merge([tf.summary.scalar('loss', val_loss_pl),
                                       tf.summary.scalar('accuracy', val_accuracy_pl)])

    # session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    print('total params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    # modele saver
    saver = tf.train.Saver(max_to_keep=100)

    if not os.path.exists(FLAGS.model_path):
        os.mkdir(FLAGS.model_path)
    else:
        shutil.rmtree(FLAGS.model_path)

    # summary writer
    if os.path.exists(FLAGS.log_dir + '/train'):
        shutil.rmtree(FLAGS.log_dir + '/train')
    if os.path.exists(FLAGS.log_dir + '/val'):
        shutil.rmtree(FLAGS.log_dir + '/val')

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')

    # start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0

    train_start_time = datetime.datetime.now()

    steps_one_epoch = FLAGS.image_num // FLAGS.batch_size

    try:
        while not coord.should_stop():
            start_time = time.time()

            # images and label
            train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
            feed_dict = {image_batch: train_images_batch, label_batch: train_labels_batch}

            # for test
            if step == 0:
                output_images(train_images_batch, './temp/')

            # get accuracy and output
            if step % 10 == 0:
                feed_dict[training_or_not] = False
                ac_res, train_summary = sess.run([accuracy, train_summary_op], feed_dict=feed_dict)
                print('Step %d: accuracy = %.4f' % (step, ac_res))
                train_writer.add_summary(train_summary, step)

            # back propagation and update params
            feed_dict[training_or_not] = True
            sess.run(train_method, feed_dict=feed_dict)

            # save model
            if step % steps_one_epoch == 0:
                saver.save(sess, FLAGS.model_path + FLAGS.model_name, global_step=step)

                n = 10
                mean_loss = 0.0
                mean_accuracy = 0.0
                for i in range(n):
                    val_images_batch, val_labels_batch = sess.run([val_images, val_labels])
                    # val_images_batch = val_images_batch[:, 14:14+227, 14:14+227, :]
                    val_images_batch = val_images_batch[:, 14:14 + 192, 14:14 + 192, :]
                    feed_dict[image_batch] = val_images_batch
                    feed_dict[label_batch] = val_labels_batch
                    feed_dict[training_or_not] = False
                    loss_val, accuracy_val = sess.run([loss, accuracy], feed_dict=feed_dict)
                    mean_loss += loss_val
                    mean_accuracy += accuracy_val

                mean_loss /= n
                mean_accuracy /= n
                val_summary = sess.run(val_summary_op, feed_dict={val_loss_pl: mean_loss,
                                                                  val_accuracy_pl: mean_accuracy})
                val_writer.add_summary(val_summary, step)

            # outupt time per step
            if step % 10 == 0:
                end_time = time.time()
                print('Time use: %.4f' % (end_time - start_time))

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        # when done, ask the threads to stop.
        coord.request_stop()
        # wait for threads to finish.
        coord.join(threads)

    sess.close()
    print('finish!')

    train_end_time = datetime.datetime.now()
    print('train start: ', train_start_time)
    print('train end: ', train_end_time)


def output_images(images, dir_path):
    for i in range(len(images)):
        img = images[i, :, :, :]
        img = img * 255
        img = img.astype('uint8')

        # cv2.imwrite(os.path.join(dir, str(i) + '_' + str(train_labels_batch[i]) + '.bmp'), img)
        cv2.imwrite(os.path.join(dir_path, str(i) + '.bmp'), img)


def test_all_saved_model():
    files = os.listdir(FLAGS.model_path)
    files = filter(lambda x: x[-4:] == 'meta', files)
    files = sorted(map(lambda x: int(x[11:-5]), files), reverse=True)
    for file in files:
        test(os.path.join(FLAGS.model_path, 'model.ckpt-'+str(file)))


def test(model_name):
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.reset_default_graph()

    # validation data batch
    val_data_reader = DataReader(FLAGS.val_data_dir, FLAGS.val_batch_size, 1, False)
    val_images, val_labels = val_data_reader.input()

    # session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # restore model
    saver = tf.train.import_meta_graph(model_name+'.meta')
    saver.restore(sess, model_name)

    graph = tf.get_default_graph()
    image_batch = graph.get_tensor_by_name('image_batch:0')
    label_batch = graph.get_tensor_by_name('label_batch:0')
    training_or_not = graph.get_tensor_by_name('training_or_not:0')
    predict_label = graph.get_tensor_by_name('softmax:0')

    right_total = 0
    data_total = 0

    # start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():

            # image and label
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

            # if using multi crop to cal accuracy
            if not FLAGS.multi_crop:
                image = val_images_batch[:, 14:14+192, 14:14+192, :]
                result = sess.run(predict_label, feed_dict={image_batch: image,
                                                            label_batch: val_labels_batch,
                                                            training_or_not: False})
                label = np.argmax(result, axis=1)
            else:
                label = 0
                croped_images_batch = multi_crop(val_images_batch)
                for images_batch in croped_images_batch:
                    result = sess.run(predict_label, feed_dict={image_batch: images_batch,
                                                                label_batch: val_labels_batch,
                                                                training_or_not: False})
                    label += result
                label = np.argmax(label, axis=1)

            right = np.sum(np.equal(label, val_labels_batch))
            right_total += right
            data_total += len(val_labels_batch)

    except tf.errors.OutOfRangeError:
        pass
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    sess.close()
    print('model: ', model_name)
    print('accuracy: {}/{} = {}'.format(right_total, data_total, right_total/data_total))


def multi_crop(images_batch):
    """multi crop function

    5-crop: four corner and center, 5 places
    10-crop: 5-crop and its flips

    :param images_batch:
    :return: 10-crop images batch
    """
    croped_images_batch = []

    index1 = FLAGS.image_size - FLAGS.input_size
    index2 = (FLAGS.image_size - FLAGS.input_size) // 2

    # four coner and center
    begin_index = [(0, 0), (0, index1), (index1, 0), (index1, index1), (index2, index2)]

    for index in begin_index:
        croped_images_batch.append(images_batch[:, index[0]:(index[0] + FLAGS.input_size),
                                   index[1]:(index[1] + FLAGS.input_size), :])

    # flip left and right
    for i in range(5):
        croped_images_batch.append(croped_images_batch[i][:, :, ::-1, :])
    return croped_images_batch


def inference(image):
    """Predict the label of image

    :param image: numpy array
    :return:
    """
    tf.reset_default_graph()

    # session
    sess = tf.Session()

    # restore model
    model_name = os.path.join(FLAGS.model_path, 'model.ckpt-3934')
    saver = tf.train.import_meta_graph(model_name + '.meta')
    saver.restore(sess, model_name)

    # useful tensors
    graph = tf.get_default_graph()
    image_batch = graph.get_tensor_by_name('image_batch:0')
    label_batch = graph.get_tensor_by_name('label_batch:0')
    training_or_not = graph.get_tensor_by_name('training_or_not:0')
    predict_label = graph.get_tensor_by_name('softmax:0')

    # if using multi crop to cal accuracy
    if not FLAGS.multi_crop:
        image = cv2.resize(image, (FLAGS.input_size, FLAGS.input_size))
        image = np.expand_dims(image, axis=0)
        result = sess.run(predict_label, feed_dict={image_batch: image,
                                                    label_batch: None,
                                                    training_or_not: False})
        label = np.argmax(result, axis=1)
    else:
        image = cv2.resize(image, (FLAGS.image_size, FLAGS.image_size))
        image = np.expand_dims(image, axis=0)

        label = 0
        croped_images_batch = multi_crop(image)
        for images_batch in croped_images_batch:
            result = sess.run(predict_label, feed_dict={image_batch: images_batch,
                                                        label_batch: None,
                                                        training_or_not: False})
            label += result
        label = np.argmax(label, axis=1)
    sess.close()
    print('label is ', label)


def main(_):
    if FLAGS.run_type == 0:
        train()
    elif FLAGS.run_type == 1:
        test_all_saved_model()
    elif FLAGS.run_type == 2:
        model_name = ''
        test(model_name)
    else:
        image = cv2.imread(FLAGS.test_pic)
        inference(image)


if __name__ == '__main__':
    tf.app.run()
