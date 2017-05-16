#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:13:18 2017

@author: josip
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os


image_size = 28
num_labels = 10
p_data='model_dataset'
data_root='/home/josip/Documents/Udacity_machine_learning/'


def load_data(pickle_file):
    _pickle=os.path.join(data_root,pickle_file)
    with open(_pickle, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
    return train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels

def reformat_data(data,labels):
    data=data.reshape((-1,image_size*image_size)).astype(np.float32)
    labels=(np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return data,labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def nn(train_d,train_l,valid_d,valid_l,test_d,test_l):
    max_iter=800
    graph=tf.Graph()
    with graph.as_default():
        tf_train_dataset=tf.constant(train_d)
        tf_train_labels=tf.constant(train_l)
        tf_test_dataset=tf.constant(test_d)
        tf_test_labels=tf.constant(test_l)
        tf_valid_dataset=tf.constant(valid_d)
        tf_valid_labels=tf.constant(valid_l)

        w=tf.Variable(initial_value=tf.truncated_normal(shape=(image_size*image_size,num_labels)))
        b=tf.Variable(tf.zeros(num_labels))
        
        logits=tf.matmul(tf_train_dataset,w)+b
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
        
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
        
        train_prediction=tf.nn.softmax(logits=logits)
        
    with tf.Session(graph=graph) as s:
        tf.global_variables_initializer().run()
        print("initialize all variable")
        for i in range(max_iter):
            _,l=s.run([optimizer,loss])
            if (i%100==0):
                 print('Loss at step %d: %f' % (i, l))
                 #print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))
                
                



def nn_batch(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    max_iter=3000
    batch_size=100
    graph=tf.Graph()
    with graph.as_default():
        tf_train_data=tf.placeholder(dtype=tf.float32,shape=(batch_size,image_size*image_size))
        tf_train_labels=tf.placeholder(dtype=tf.int32, shape=(batch_size,num_labels))
        
        w=tf.Variable(initial_value=tf.truncated_normal(shape=(image_size*image_size,num_labels)))
        b=tf.Variable(initial_value=tf.zeros(shape=(num_labels)))
        
        logits=tf.matmul(tf_train_data,w)+b
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
        opt=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=loss)
        
        
    with tf.Session(graph=graph) as sess:
       tf.global_variables_initializer().run()
       for i in range(max_iter):
           offset=(i* batch_size)%(train_dataset.shape[0]-batch_size)
           batch_data = train_dataset[offset:(offset + batch_size), :]
           batch_labels = train_labels[offset:(offset + batch_size),:]
           feed={tf_train_data:batch_data,tf_train_labels:batch_labels}
           
           _,l=sess.run([opt,loss],feed_dict=feed)

           if(i%400==0):
               print("error:",i,l)
           


def problem_1(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    max_iter=3000
    batch_size=100
    hidden_layer_neuron=1024
    
    graph=tf.Graph()
    
    with graph.as_default():
        
        tf_input=tf.placeholder(dtype=tf.float32,shape=(batch_size,image_size*image_size))
        tf_labels=tf.placeholder(dtype=tf.float32,shape=(batch_size,num_labels))
        
        tf_test=tf.constant(value=test_dataset)
        
        #w_1, b_1 = input to hidden
        w_1= tf.Variable(initial_value=tf.truncated_normal(shape=(image_size*image_size,hidden_layer_neuron)))
        b_1=tf.Variable(initial_value=tf.zeros(hidden_layer_neuron))
        
        #w_2, b_2 = hidden to output
        w_2=tf.Variable(initial_value=tf.truncated_normal(shape=(hidden_layer_neuron,num_labels)))
        b_2=tf.Variable(initial_value=tf.zeros(num_labels))
        
        logits=tf.matmul(tf.nn.relu(tf.matmul(tf_input,w_1)+b_1),w_2)+b_2
        
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_labels))
        
        opt= tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=loss)
        
        tf_test_logits=tf.matmul(tf.nn.relu(tf.matmul(tf_test,w_1)+b_1), w_2) + b_2
        tf_test_prediction=tf.nn.softmax(logits=tf_test_logits)
                
    
    with tf.Session(graph=graph) as sess:
        
        tf.global_variables_initializer().run()
        
        for i in range(max_iter):
           offset=(i* batch_size)%(train_dataset.shape[0]-batch_size)
           batch_data = train_dataset[offset:(offset + batch_size), :]
           batch_labels = train_labels[offset:(offset + batch_size),:]
           feed={tf_input:batch_data,tf_labels:batch_labels}
           
           _,l=sess.run([opt,loss],feed_dict=feed)

           if(i%400==0):
               print("error:",i,l)
        
        print("Test acc:", accuracy(tf_test_prediction.eval(),test_labels))

train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels=load_data(p_data)

train_dataset, train_labels = reformat_data(train_dataset, train_labels)
valid_dataset, valid_labels = reformat_data(valid_dataset, valid_labels)
test_dataset, test_labels = reformat_data(test_dataset, test_labels)


problem_1(train_dataset, train_labels,valid_dataset, valid_labels,test_dataset, test_labels)