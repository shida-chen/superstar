#!/usr/bin/python
#-*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import sys  
from tensorflow.examples.tutorials.mnist import input_data 
 
def weight_variable(shape):  
    return tf.Variable( tf.truncated_normal(shape, stddev=0.1) )  
def bias_variable(shape):  
    return tf.Variable( tf.constant(0.1, shape=shape) )  
def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):  
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 随机取batch个训练样本
def next_batch(train_data, train_target, batch_size):
    idx = [ i for i in range(0,len(train_target)) ]
    np.random.shuffle(idx);
    batch_data = []; batch_target = [];
    for i in range(0,batch_size):
        batch_data.append(train_data[idx[i]]);
        batch_target.append(train_target[idx[i]])
    return batch_data, batch_target
"""   初始化参数    """
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #注意存放数据的路径
train_data = mnist.train.images  #55000的数据量
train_target = mnist.train.labels
test_data = mnist.test.images    #10000的数据量
test_target = mnist.test.labels
x = tf.placeholder("float", shape=[None, 784])  #训练向量
y = tf.placeholder("float", shape=[None, 10])   #真实结果
keep_prob = tf.placeholder("float") # keep_probability 隐含层节点保持工作的概率
epochs_num = 5000 #训练次数
batch_size = 100 #分批次大小
"""   创建CNN第一卷积层     """
# 定义卷积核的大小5*5,传入一个图像，32个卷积核，所以传出32个图像。
W_conv1 = weight_variable([5, 5, 1, 32])  
b_conv1 = bias_variable([32])  # 定义bias的大小，为卷积核的个数
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 图片变成标准网络的输入参数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 使用relu激活函数
h_pool1 = max_pool_2x2(h_conv1)   # 输出[14,14,32,64]
"""   创建CNN第二卷积层     """
W_conv2 = weight_variable([5, 5, 32, 64])  
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu( conv2d(h_pool1, W_conv2) + b_conv2 )  
h_pool2 = max_pool_2x2(h_conv2) # 输出[7,7,64,1024]
"""   创建CNN第一全连接层     """ 
W_fc1 = weight_variable([7 * 7 * 64, 1024])  
b_fc1 = bias_variable([1024])  
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) + b_fc1 )   
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  #某些隐含层节点的权重不工作
"""   创建CNN第二全连接层     """ 
W_fc2 = weight_variable([1024, 10])  
b_fc2 = bias_variable([10])  
y_conv=tf.nn.softmax( tf.matmul(h_fc1_drop, W_fc2) + b_fc2 )  
"""   session     """
sess = tf.InteractiveSession()  
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  
sess.run(tf.initialize_all_variables())  
for i in range(epochs_num):  
    batch_data, batch_target = next_batch(train_data,train_target,batch_size)  
    if i%100 == 0:  
        train_accuracy = accuracy.eval(feed_dict={ x:batch_data, y: batch_target, keep_prob: 1.0} )  
        print ("step {0:d}, training accuracy {1:.3f}".format(i, train_accuracy))
    train_step.run(feed_dict={x: batch_data, y:batch_target, keep_prob: 0.5})  
 
print ("Training finished") 
print ("test accuracy {0:.3f}".format(accuracy.eval(feed_dict={ x: test_data, y:test_target , keep_prob: 1.0})))
