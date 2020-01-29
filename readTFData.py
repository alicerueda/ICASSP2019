#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:24:20 2018

@author: arueda
"""

#import os
#import sys
#import glob
import tensorflow as tf
#import numpy as np
#import pandas as pd
#from random import shuffle

DEBUG = True
BASIC = True
HC = False
MIDSECTION = 12500
path = '/home/arueda/SaarbruckenVoiceDB/'
#path = '/Volumes/AliceSSD500/SaarbruckenVoiceDB/
csv_path = path +'FSST/Normalized/'
Sound_Types = ['a_n', 'a_h', 'a_l', 'a_lhl', 'i_n', 'i_h', 'i_l','i_lhl', 'u_n', 'u_h', 'u_l','u_lhl']
CATEGORIES = ['HC','Dys'] 
#out_filename = path + '/PythonCode/Dataset/'  #The filename will be changed to FSSTSubjectTypeSoundType.tfrecords

#category = CATEGORIES[1]
sound_type = Sound_Types[0]

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

def input_data(filename,batchsize):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(decode)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batchsize)
    #iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    return iterator


TEST = False
if TEST:
    out_filename = path + '/PythonCode/Dataset/DataAug'+sound_type+'_test.tfrecords'
    batchsize =20
else:
    out_filename = path + '/PythonCode/Dataset/DataAug'+sound_type+'_train.tfrecords'
    batchsize =168*10


iterator = input_data(out_filename, batchsize)
# Start a session to see how the batch load is working
sess = tf.InteractiveSession()

next_element = iterator.get_next()  #getting next batch
#print(sess.run(next_element))
#print(sess.run(next_element[0][0]))  # sqfsst_batch = next_element[0][:]
print(sess.run(next_element[1]))     # label_batch
print(sess.run(next_element[2]))     # ID batch
print(sess.run(next_element[3]))     # sex_batch
print(sess.run(next_element[4]))     # age_batch

#sess.close()
