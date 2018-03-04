import tensorflow as tf
import cv2
import numpy as np
import os

base_dir = '/home/hust/genderAndAge/data'
files_list = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt']

img_size = 227

# get all data from txt files
all_data = []
for txt in files_list:
    with open(os.path.join(base_dir, txt), 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            data = line.strip().split('\t')
            all_data.append([data[0], data[2]+'.'+data[1], data[3], data[4]])

# class of age and gender
age_class = {'(0, 2)': 0,
             '(4, 6)': 1,
             '(8, 12)': 2,
             '(15, 20)': 3,
             '(25, 32)': 4,
             '(38, 43)': 5,
             '(48, 53)': 6,
             '(60, 100)': 7}
gender_class = {'m': 0, 'f': 1}

age_data = []
gender_data = []
age_gender_data = []

prefix = ''
for data in all_data:
    try:
        if data[2] == '(38, 42)' or data[2] == '(38, 48)':
            data[2] = '(38, 43)'

        if data[2] == '(27, 32)':
            data[2] = '(25, 32)'

        if data[2] not in age_class and data[2] != 'None':
            age = int(data[2])
            if 0 <= age <= 3:
                data[2] = '(0, 2)'
            elif 4 <= age <= 7:
                data[2] = '(4, 6)'
            elif 8 <= age <= 14:
                data[2] = '(8, 12)'
            elif 15 <= age <= 24:
                data[2] = '(15, 20)'
            elif 25 <= age <= 37:
                data[2] = '(25, 32)'
            elif 38 <= age <= 47:
                data[2] = '(38, 43)'
            elif 48 <= age <= 59:
                data[2] = '(48, 53)'
            elif 60 <= age <= 100:
                data[2] = '(60, 100)'

        if data[2] != 'None' and data[3] != 'u':
            age_gender_data.append((os.path.join(base_dir, 'aligned/' + data[0] + '/landmark_aligned_face.' + data[1]),
                                    age_class[data[2]], gender_class[data[3]]))
            gender_data.append((os.path.join(base_dir, 'aligned/' + data[0] + '/landmark_aligned_face.' + data[1]),
                                gender_class[data[3]]))
            age_data.append((os.path.join(base_dir, 'aligned/' + data[0] + '/landmark_aligned_face.' + data[1]),
                             age_class[data[2]]))
        elif data[2] == 'None' and data[3] != 'u':
            gender_data.append((os.path.join(base_dir, 'aligned/' + data[0] + '/landmark_aligned_face.' + data[1]),
                                gender_class[data[3]]))
        elif data[2] != 'None' and data[3] == 'u':
            age_data.append((os.path.join(base_dir, 'aligned/' + data[0] + '/landmark_aligned_face.' + data[1]),
                             age_class[data[2]]))
    except Exception as e:
        print(data)

print(len(age_data))
print(len(gender_data))
print(len(age_gender_data))


def make_age_data(data):

    count = 0
    num_file = 2000
    i = 0
    while True:
        writer = tf.python_io.TFRecordWriter('age' + str(i) + '.tfrecords')
        for j in range(num_file):
            if count == len(data):
                break
            label = data[count][1]

            img = cv2.imread(data[count][0])
            img = cv2.resize(img, (img_size, img_size))
            if img is not None:
                image_raw = img.tostring()
                feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                           'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            count += 1
        i += 1
        if count == len(data):
            break


def make_gender_data(data):

    count = 0
    num_file = 2000
    i = 0
    while True:
        writer = tf.python_io.TFRecordWriter('gender' + str(i) + '.tfrecords')
        for j in range(num_file):
            if count == len(data):
                break
            label = data[count][1]

            img = cv2.imread(data[count][0])
            img = cv2.resize(img, (img_size, img_size))
            if img is not None:
                image_raw = img.tostring()
                feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                           'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            count += 1
        i += 1
        if count == len(data):
            break


def make_age_gender_data(data):

    count = 0
    num_file = 2000
    i = 0
    while True:
        writer = tf.python_io.TFRecordWriter('age_gender' + str(i) + '.tfrecords')
        for j in range(num_file):
            if count == len(data):
                break
            age = data[count][1]
            gender = data[count][2]

            img = cv2.imread(data[count][0])
            img = cv2.resize(img, (img_size, img_size))
            if img is not None:
                image_raw = img.tostring()
                feature = {'age': tf.train.Feature(int64_list=tf.train.Int64List(value=[age])),
                           'gender': tf.train.Feature(int64_list=tf.train.Int64List(value=[gender])),
                           'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            count += 1
        i += 1
        if count == len(data):
            break


make_gender_data(gender_data)
# make_age_data(gender_data)
