#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:56:35 2017

@author: josip
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os

image_size = 28
num_labels = 10
p_data='model_dataset'
data_root='/home/josip/Documents/Udacity_machine_learning/'

def load_dataset(pickle_file):
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
    return train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels


def reformat_data(data,labels):
    data=data.reshape((-1,image_size*image_size)).astype(np.float32)
    labels=(np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return data,labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / labels.shape[0])
  
def get_1_layer_graph(x_size,f_size,y_size,t_size):
    l2_beta=0.05
    g=tf.Graph()
    with g.as_default():
        
        tf_input=tf.placeholder(dtype=tf.float32,shape=(x_size,f_size*f_size),name='tf_input')
        tf_labels=tf.placeholder(dtype=tf.float32,shape=(x_size,y_size),name='tf_labels')
        
        tf_tst=tf.placeholder(dtype=tf.float32,shape=(t_size,f_size*f_size),name='tf_tst')
        
        w=tf.Variable(initial_value=tf.truncated_normal(shape=(f_size*f_size,y_size)))
        b=tf.Variable(initial_value=tf.zeros(y_size))
        
        logits=tf.matmul(tf_input,w)+b
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_labels),name='loss')
        
        w_req=tf.nn.l2_loss(w)
        loss=loss+l2_beta*w_req
        opt=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss,name='opt')
            
        tf_tst_prediction=tf.nn.softmax(tf.matmul(tf_tst,w)+b,name='test_prediction')
        
        
    
    return g


def get_2_layer_graph(x_size,f_size,y_size,t_size,h_size):
    l2_beta=0.05
    g=tf.Graph()
    with g.as_default():
         tf_input=tf.placeholder(dtype=tf.float32,shape=(x_size,f_size*f_size),name='tf_input')
         tf_labels=tf.placeholder(dtype=tf.float32,shape=(x_size,y_size),name='tf_labels')
         
         tf_tst=tf.placeholder(dtype=tf.float32,shape=(t_size,f_size*f_size),name='tf_tst')
         
         w_1=tf.Variable(initial_value=tf.truncated_normal(shape=(f_size*f_size,h_size)))
         b_1=tf.Variable(initial_value=tf.zeros(shape=(h_size)))
         
         w_2=tf.Variable(initial_value=tf.truncated_normal(shape=(h_size,y_size)))
         b_2=tf.Variable(initial_value=tf.zeros(shape=(y_size)))
      
         l_1=tf.nn.relu(tf.matmul(tf_input,w_1)+b_1)
         
         tf_output=tf.nn.softmax_cross_entropy_with_logits(logits=(tf.matmul(l_1,w_2)+b_2),labels=tf_labels)
         
         loss=tf.reduce_mean(tf_output,name='loss')
         loss=loss+l2_beta*tf.nn.l2_loss(w_1)+l2_beta*tf.nn.l2_loss(w_2)
         opt=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss=loss,name='opt')
         
         tf_tst_prediction=tf.nn.softmax(
                 logits=tf.matmul(tf.nn.relu(tf.matmul(tf_tst,w_1)+b_1),w_2)+b_2,name='test_prediction')
         
         
    return g

def problem_1(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,nn_2_layer=False):
    max_iter=3000
    batch_size=200
    hidden_layer_size=image_size*image_size
    if nn_2_layer:
        g=get_2_layer_graph(batch_size,image_size,num_labels,test_dataset.shape[0],hidden_layer_size)
    else:
        g=get_1_layer_graph(batch_size,image_size,num_labels,test_dataset.shape[0])

    with tf.Session(graph=g) as s:
        
        tf.global_variables_initializer().run()
        
        for step in range(max_iter):
            
            tf_input=g.get_tensor_by_name(name='tf_input:0')
            tf_labels=g.get_tensor_by_name(name='tf_labels:0')
            opt=g.get_operation_by_name(name='opt')
            loss=g.get_tensor_by_name(name='loss:0')
            tf_tst_prediction=g.get_tensor_by_name(name="test_prediction:0")
            tf_tst=g.get_tensor_by_name(name='tf_tst:0')
            
            offset=(step*batch_size)%(train_dataset.shape[0]-batch_size)
            data_b=train_dataset[offset:(offset+batch_size),:]
            labels_b=train_labels[offset:(offset+batch_size),:]
            d={tf_input:data_b,tf_labels:labels_b}
            _,l=s.run([opt,loss],feed_dict=d)
            
            if(step % 500 == 0):
                print("loss:",l)
       
        print("tst acc",accuracy(tf_tst_prediction.eval(feed_dict={tf_tst:test_dataset}),test_labels))
    

train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels=load_dataset(p_data)

train_dataset, train_labels = reformat_data(train_dataset, train_labels)
valid_dataset, valid_labels = reformat_data(valid_dataset, valid_labels)
test_dataset, test_labels = reformat_data(test_dataset, test_labels)

problem_1(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,True)