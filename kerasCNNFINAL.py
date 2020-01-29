#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:42:20 2018

@author: arueda
"""
import tensorflow as tf
import os
tf.enable_eager_execution()
#tf.executing_eagerly()
i = 0
DEBUG = False

# Samples inside the TFRecords
num_tfrecords = 1
samples_in_train = 168*10   # 84*2*Data Augmentation x10 times
samples_in_test = 20*10        # 10*2*10
# Load Dataset
path = '/home/arueda/SaarbruckenVoiceDB/PythonCode/Dataset'
Sound_Types = ['a_n', 'a_h', 'a_l', 'a_lhl', 'i_n', 'i_h', 'i_l','i_lhl', 'u_n', 'u_h', 'u_l','u_lhl']

os.chdir(path)
filenames_train = [
        'a_n_train.tfrecords', 'a_h_train.tfrecords', 'a_l_train.tfrecords', 'a_lhl_train.tfrecords',
        'i_n_train.tfrecords', 'i_h_train.tfrecords', 'i_l_train.tfrecords', 'i_lhl_train.tfrecords',
        'u_n_train.tfrecords', 'u_h_train.tfrecords', 'u_l_train.tfrecords', 'u_lhl_train.tfrecords']
filenames_test = [
        'a_n_test.tfrecords','a_h_test.tfrecords','a_l_test.tfrecords','a_lhl_test.tfrecords',
        'i_n_test.tfrecords','i_h_test.tfrecords','i_l_test.tfrecords','i_lhl_test.tfrecords',
        'u_n_test.tfrecords','u_h_test.tfrecords','u_l_test.tfrecords','u_lhl_test.tfrecords']

    
#Data information
num_classes = 2
image_size = 50
num_channels = 1

def decode(serialized_example):
    r_size = 50
    c_size = 50
        
    # Define feature
    read_features = {'sqfsst': tf.FixedLenFeature([],tf.string),
                     'subjectID': tf.FixedLenFeature([], tf.int64),
                     'label': tf.FixedLenFeature([], tf.int64),
                     #'voicetype': tf.VarLenFeature(dtype=tf.string),
                     'age': tf.FixedLenFeature([], tf.int64),
                     'sex': tf.FixedLenFeature([], tf.int64)}

    # Decode the record read by the reader
    parse_example = tf.parse_single_example(serialized=serialized_example, features=read_features)

    sqfsst = tf.decode_raw(parse_example['sqfsst'], tf.float32)
    sqfsst = tf.cast(sqfsst, tf.float32)
    sqfsst = tf.reshape(sqfsst, [r_size,c_size])
    label = tf.cast(parse_example['label'], tf.int64)
    subjectID = tf.cast(parse_example['subjectID'],tf.int64)
    sex = tf.cast(parse_example['sex'], tf.int64)
    age = tf.cast(parse_example['age'], tf.int64)
    return sqfsst, label, subjectID, sex, age


def input_data(filename,batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(decode)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(4096)
    dataset = dataset.batch(batch_size)
    #iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    return iterator


# Start a session to see how the batch load is working
if DEBUG:
    batch_size_train = samples_in_train
    batch_size_test = samples_in_test
    iterator_train = input_data(filenames_train, batch_size_train)
    iterator_test = input_data(filenames_test,batch_size_test)
    Train_Images, Train_Labels,_,_,_ = iterator_train.get_next()
    Test_Images, Test_Labels,_,_,_ = iterator_test.get_next()    
    Train_Images = tf.reshape(Train_Images,[-1, image_size,image_size,1])
    #Train_Labels = tf.one_hot(Train_Labels,num_classes)
    Test_Images = tf.reshape(Test_Images,[-1, image_size,image_size,1])
    #Test_Labels = tf.one_hot(Test_Labels,num_classes)
    #sess = tf.InteractiveSession()    #Session is needed when not running eager execution
    print(Train_Images)       #without eager execution, you need to run inside a session: print(sess.run(Train_Images))
    print(Train_Labels)
    print(Test_Images)
    print(Test_Labels)
    print(Train_Images)
    print(Train_Labels)
    
#from tensorflow.python import keras as keras
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout#, BatchNormalization
#from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers #Optimizer has to be from tensorflow not keras in eager mode

#bn=BatchNormalization()

def create_model():
    model = Sequential()
   #model.add(BatchNormalization(axis=1))
    model.add(Conv2D(8,(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(image_size,image_size,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    #model.add(Dropout(rate=0.5))
    model.add(Conv2D(8,(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(image_size,image_size,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    #model.add(Dropout(rate=0.5))
    #model.add(Conv2D(64, (5,5), strides=(1,1),activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    #model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(0.001), activation='softmax'))
    #model.add(Dense(num_classes, kernel_initializer='orthogonal',activation='softmax'))
    return model

model = create_model()
EPOCHS=200
learning_rate = 0.01
model.compile(optimizer=tf.train.AdagradOptimizer(learning_rate, name='Adagrad'),#.optimizers.Adagrad(lr=learning_rate),
              loss='binary_crossentropy',#'categorical_crossentropy', #loss='mse',
              metrics=['acc','mse'])
#model.summary()

sound_type = Sound_Types[i]
train_file = filenames_train[i]
test_file = filenames_test[i]
# Training Data
batch_size = samples_in_train * num_tfrecords
iterator_train = input_data(train_file,batch_size)
Train_Images, Train_Labels,_,_,_ = iterator_train.get_next()
Train_Images = tf.reshape(Train_Images,[batch_size, image_size,image_size,1])
Train_Labels = tf.one_hot(Train_Labels,num_classes)
# Test Data
batch_size = samples_in_test * num_tfrecords
iterator_test = input_data(test_file,batch_size)
Test_Images, Test_Labels,_,_,_ = iterator_test.get_next()    
Test_Images = tf.reshape(Test_Images,[batch_size, image_size,image_size,1])
Test_Labels = tf.one_hot(Test_Labels,num_classes)

history = model.fit(x=Train_Images,
                    y=Train_Labels,
                    batch_size=15,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(Test_Images,Test_Labels))

score = model.evaluate(Train_Images, Train_Labels,verbose=1)
if DEBUG:
    print('CNN Error: %.2f%%' %(100-score[1]*100))
    print(model.metrics_names)

# Testing model
eval_batch_size = 5
model.evaluate(Test_Images, Test_Labels, eval_batch_size, verbose =1)    
predict =model.predict(Test_Images, eval_batch_size, verbose =1)

if DEBUG: 
    print(predict)  #Predict returns probability. We need to convert round this to get 0 or 1

# Writing history to file
import pandas as pd
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
val_mean_squared_error = history.history['val_mean_squared_error']
loss = history.history['loss']
acc = history.history['acc']
mean_squared_error = history.history['mean_squared_error']
measure_header = ['val_loss', 'val_acc', 'val_mean_squared_error', 'loss', 'acc', 'mean_squared_error']
filename = '../CNNPerformance/CNNMeasure'+sound_type+'.csv'
perf_measure = [val_loss, val_acc, val_mean_squared_error, loss, acc, mean_squared_error]
pd.DataFrame(perf_measure).to_csv(filename, index=measure_header, header=False)

# Calculate Conf. Matrix
import numpy as np
predict = tf.round(predict)
predictions = tf.where(tf.equal(predict,1))
true_labels = tf.where(tf.equal(Test_Labels,1))
conf_mat = tf.confusion_matrix(true_labels[:,1], predictions[:,1],2)
conf_mat = np.asarray(conf_mat)
filename = '../CNNPerformance/CNNConfMatrix'+sound_type+'.csv'
pd.DataFrame(conf_mat).to_csv(filename, index=False, header=False)


print('The confusion matrix is:')
print(conf_mat)    
print('True Labels:',true_labels[:,1])
print('Preditions:', predictions[:,1])

import matplotlib.pyplot as plt
# Plotting Results
print(history.history.keys())

txt = 'Loss Function for ' + sound_type +' FSST'
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(txt)
plt.legend(['Train Loss', 'Test Loss'],loc='lower left')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

txt = 'Model accuracy for ' + sound_type + ' FSST'
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(txt)
plt.legend(['Train Acc','Test Acc'],loc='upper left')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

