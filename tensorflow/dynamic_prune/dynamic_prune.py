import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import read_image
from tensorflow.examples.tutorials.mnist import input_data
sys.dont_write_bytecode = True

argparser = argparse.ArgumentParser()
argparser.add_argument("-1", "--train", action="store_true", help="train dense MNIST model")
argparser.add_argument("-0", "--check", action="store_true", help="check for time")
argparser.add_argument("-2", "--prune", action="store_true", help="prune model and retrain or iteratively")
argparser.add_argument("-3", "--recover", action= "store_true", help="recover dense model and retrain")
argparser.add_argument("-t", "--train_iterations", default="20000", help="pre-train dense model with t iterations")
argparser.add_argument("-m", "--dense", default="./models/model_ckpt_dense", help="Target checkpoint model file for 2nd round")
argparser.add_argument("-n", "--sparse",default="./models/model_ckpt_pruned_retrained", help="Target checkpoit model file for 3rd round")
argparser.add_argument("-i", "--iteration", default="1", help="iteration number")
argparser.add_argument("-p", "--percentage", default="0.6", help="pruning percentage in every iteration")
argparser.add_argument("-a", "--pre_v", default="100", help="pre-training times")
argparser.add_argument("-b", "--pru_v", default="200", help="prune-retraning times")
argparser.add_argument("-c", "--rec_v", default="300", help="recover-tuning  times")
argparser.add_argument("-x", "--pre_acc", default="0.85", help="pre-training accurancy")
argparser.add_argument("-y", "--pru_acc", default="0.99", help="prune-retraining accurancy")
argparser.add_argument("-z", "--rec_acc", default="0.99", help="recover-tuning  times")
args = argparser.parse_args()

if(args.train or args.prune or args.recover or args.check) == False:
    argparser.print_help()
    sys.exit()

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def dense_cnn_model(image, weights, keep_prob):
    x_image = tf.reshape(image, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
    h_pool1 = max_pool_2x2(h_conv1)   #[-1,14,14,32]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
    h_pool2 = max_pool_2x2(h_conv2)   #[-1,7,7,64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights["w_fc1"]) + weights["b_fc1"])
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    logit = tf.matmul(h_fc1_dropout, weights["w_fc2"]) + weights["b_fc2"]     #[-1,10]
    return logit

def test(predict_logit):
    correct_prediction = tf.equal(tf.argmax(predict_logit,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = 0
    for i in range(20):
        batch = mnist.test.next_batch(500)
	result = result + sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob : 1.0})
    result = result / 20.0
    return result

def prune(weights, th):
    '''
    :param weights: weight Variable
    :param th: float value, weight under th will be pruned
    :return: sparse_weight: weight matrix after pruning, which is a 2d numpy array
    ~under_threshold: boolean matrix in same shape with sparse_weight indicating whether corresponding element is zero
    '''
    weight_array = sess.run(weights)
    under_threshold = abs(weight_array) < th
    weight_array[under_threshold] = 0
    return weight_array, ~under_threshold

def get_th(weight, percentage=0.8):
    flat = tf.reshape(weight, [-1])
    flat_list = sorted(map(abs,sess.run(flat)))
    return flat_list[int(len(flat_list) * percentage)]

def delete_none_grads(grads):
    count = 0
    length = len(grads)
    while(count < length): 
        if(grads[count][0] == None):
	    del grads[count]
	    length -= 1
	else:
	    count += 1

mnist = input_data.read_data_sets('./data/', one_hot=True)
file_object = open("log.txt", "w")
sess = tf.Session()

dense_w = {
    "w_conv1":tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1), name="w_conv1"),
    "b_conv1":tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1"),
    "w_conv2":tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name="w_conv2"),
    "b_conv2":tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2"),
    "w_fc1":tf.Variable(tf.truncated_normal([7*7*64,1024], stddev=0.1), name="w_fc1"),
    "b_fc1":tf.Variable(tf.constant(0.1, shape=[1024]), name="b_fc1"),
    "w_fc2":tf.Variable(tf.truncated_normal([1024,10], stddev=0.1), name="w_fc2"),
    "b_fc2":tf.Variable(tf.constant(0.1, shape=[10]), name="b_fc2")
}

counter = 0
accurancy = 0
# The First Phase for Training
if(args.train == True):
    t_pt_0 = time.clock()
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
   
    logit = dense_cnn_model(x, dense_w, keep_prob)
   
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.arg_max(logit,1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
	init = tf.global_variables_initializer()
    sess.run(init)
   
    file_object.write("\n [Info]---------------------------Pre-training Result List----------------------[Report]\n")
    print("\033[0;31m [Info]--------------Pre-training Start--------------------[Report]\033[0m")
    for i in range(int(args.train_iterations)):
        batch = mnist.train.next_batch(50)
	
        train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
	if train_acc >= float(args.pre_acc):
	    counter = counter + 1
	    accurancy = accurancy + train_acc
  
	if counter == int(args.pre_v):
	    print(" Step = %d, training accuramcy = %g" % (i, train_acc))
            print("[Info] Pre-training accurancy has became stable with Accurancy = %g" % (accurancy/counter))
	    file_object.write("[Info] Pre-training accurancy has became stable with Accurancy = %g \n" % (accurancy/counter))
            break

	if i % 1000 == 0:
            train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
	    print("[Info] Step = %d, training accuracy = %g" % (i, train_acc))
        sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        
    t_pt_1 = time.clock()
    print("[Info] Pretrain Phase Time :\n Pre-training = " + str((t_pt_1 - t_pt_0)*1000) + " ms \n")
    print("\033[0;31m [Info]--------------Pre-training End--------------------[Report]\033[0m")
    print("\033[0;31m [Info]-----------------Test Start--------------------[Report]\033[0m")
    start = time.clock()
    test_acc = test(logit)
    end = time.clock()
    print("\nDense model:test accuracy = %g" % test_acc + ", inference time = " + str((end - start)*1000) + "ms \n")
    file_object.write("[Info] Pre-train Phase Time :\n Pre-training = " + str((t_pt_1 - t_pt_0)*1000) + "ms \n")
    file_object.write("Dense model:test accuracy = %g" % test_acc +  ", inference time = " + str((end - start)*1000) + "ms \n")
    print("\033[0;31m [Info]-----------------Test End---------------------[Report]\033[0m")

    saver = tf.train.Saver()
    saver.save(sess, "./models/model_ckpt_dense")
    print("[Info] Successfully saved model_ckpt_dense!")

# The Checking phase for training
counter = 0
accurancy = 0
if(args.check == True):
    print("\033[0;31m [Info]---------------------check Start-------------------[Report]\033[0m")
    t_pr_0 = time.clock()
    th = time.clock()
    saver = tf.train.Saver()
    saver.restore(sess, args.dense)
    te = time.clock()
    print("\n $$$$$$$$$$$$ time checking0 = " + str((te - th)*1000)+"ms $$$$$$$$$$ \n")
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    logit = dense_cnn_model(x, dense_w, keep_prob)

    start = time.clock()
    test_acc = test(logit)
    end = time.clock()
    th = time.clock()
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
    trainer = tf.train.AdamOptimizer(1e-4)
    grads = trainer.compute_gradients(cross_entropy)

    train_step = trainer.apply_gradients(grads)

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1: 
        initial = tf.all_variables()
    else:
	initial = tf.global_variables()
    for var in initial:
	if sess.run(tf.is_variable_initialized(var)) == False:
            sess.run(var.initializer)
    te = time.clock()
    print("\n $$$$$$$$$$$$$ time checking2 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")

    print("\033[0;31m [Info]--------------Retraining Start--------------------[Report]\033[0m")
    th = time.clock()
    for j in range(int(args.train_iterations)):
        batch = mnist.train.next_batch(50)

	train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
	if train_acc >= float(args.pru_acc):
	    counter = counter + 1
	    accurancy = accurancy + train_acc
        if counter == int(args.pru_v):
            print("Retraining step %d, accurancy %g" % (j, train_acc))
            print("[Info] Training accurancy has became stable with Accurancy = %g" % (accurancy/counter))
	    break
        if (j % 1000 == 0):
	    train_acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print("Retraining step %d, accurancy %g" % (j, train_acc))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    te = time.clock()
    print("\n $$$$$$$$$$$$$ time checking3 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")

    t_pr_1 = time.clock()
    print("[Info] checking Phase Time :\n Pruning = " + str((t_pr_1 - t_pr_0)*1000) + " ms \n")
    start = time.clock()
    test_acc = test(logit)
    end = time.clock()

    print("\033[0;31m Retraining: Iteration = %d , Test accurancy after pruning and retraining = %g , Inference time = %g ms \033[0m" % (i, test_acc,(end - start)*1000))
    print("\033[0;31m [Info]--------------Retraining End--------------------[Report]\033[0m")

    saver = tf.train.Saver(dense_w)
    saver.save(sess, "./models/model_ckpt_checking")
    print("[Info] Successfully saved model_ckpt_checking !")

# The Sencond Phase for Pruning and Retraining
counter = 0
accurancy = 0
if(args.prune == True):
    print("\033[0;31m [Info]---------------------Pruning Start-------------------[Report]\033[0m")
    print("Total pruning iteration: %d. pruning percentage each iter: %g" % (int(args.iteration), float(args.percentage)))
    file_object.write("\n [Info]-------------------------Pruning and Retraining Result List--------------------[Report]\n")
    file_object.write("Total pruning iteration: %d. pruning percentage each iteration: %g\n" % (int(args.iteration), float(args.percentage)))
    t_pr_0 = time.clock()
    th = time.clock()
    saver = tf.train.Saver()
    saver.restore(sess, args.dense)
    te = time.clock()
    print("\n $$$$$$$$$$$$ time checking0 = " + str((te - th)*1000)+"ms $$$$$$$$$$ \n")
    th = time.clock()
    p = 1.0
    for i in range(int(args.iteration)):
        p = p * float(args.percentage)
	print("\033[0;31m pruning : iter %d, perc=%g \033[0m" % (i, p))
	file_object.write("pruning : iter %d, perc=%g \n" % (i, p))
	file_object.flush()
	th_fc1 = get_th(dense_w["w_fc1"], percentage=(1.0 - p))
	th_fc2 = get_th(dense_w["w_fc2"], percentage=(1.0 - p))
	sparse_w_fc1, idx_fc1 = prune(dense_w["w_fc1"], th_fc1)
	sparse_w_fc2, idx_fc2 = prune(dense_w["w_fc2"], th_fc2)
        sess.run(tf.assign(dense_w["w_fc1"], sparse_w_fc1))
        sess.run(tf.assign(dense_w["w_fc2"], sparse_w_fc2))

        array_w_fc1 = sess.run(dense_w["w_fc1"])
        array_w_fc2 = sess.run(dense_w["w_fc2"])

        print("none-zero in fc1 :%d / %d" % (np.sum(array_w_fc1 != 0),np.size(array_w_fc1)))
        print("none-zero in fc2 :%d / %d" % (np.sum(array_w_fc2 != 0),np.size(array_w_fc2)))

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            initial = tf.all_variables()
        else:
            initial = tf.global_variables()
    	for var in initial:
            if sess.run(tf.is_variable_initialized(var)) == False:
                sess.run(var.initializer)

     	x = tf.placeholder(tf.float32, [None, 784], name="x")
    	y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    	keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        logit = dense_cnn_model(x, dense_w, keep_prob)

    	start = time.clock()
    	test_acc = test(logit)
        end = time.clock()
        te = time.clock()
        print("\n $$$$$$$$$$$$$ time checking1 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")
    	print("\033[0;31m Pruning : Iteration = %d , Test accurancy after pruning = %g , Inference time = %g ms \033[0m" % (i, test_acc,(end - start)*1000))
    	print("\033[0;31m [Info]--------------Pruning End--------------------[Report]\033[0m")
    	file_object.write("Pruning : Iteration = %d , Test accurancy after pruning = %g , Inference time = %g ms\n" % (i, test_acc,(end - start)*1000))
        th = time.clock()
    	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
    	trainer = tf.train.AdamOptimizer(1e-4)
    	grads = trainer.compute_gradients(cross_entropy)
        delete_none_grads(grads)

    	count = 0
    	for grad, var in grads:
            if (var.name == "w_fc1:0"):
                idx_in1 = tf.cast(tf.constant(idx_fc1), tf.float32)
	        # grads[count] = (tf.multiply(idx_in1, grad), var)
                grads[count] = (tf.mul(idx_in1, grad), var)
	    if (var.name == "w_fc2:0"):
                idx_in2 = tf.cast(tf.constant(idx_fc2), tf.float32)
                # grads[count] = (tf.multiply(idx_in2, grad), var)
                grads[count] = (tf.mul(idx_in2, grad), var)
	    count += 1
        train_step = trainer.apply_gradients(grads)

     	correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            initial = tf.all_variables()
	else:
	    initial = tf.global_variables()
	for var in initial:
	    if sess.run(tf.is_variable_initialized(var)) == False:
		sess.run(var.initializer)
        te = time.clock()
        print("\n $$$$$$$$$$$$$ time checking2 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")

    	print("\033[0;31m [Info]--------------Retraining Start--------------------[Report]\033[0m")
        th = time.clock()
        for j in range(int(args.train_iterations)):
            batch = mnist.train.next_batch(50)
            idx_in1_value = sess.run(idx_in1)
	    grads_fc1_value = sess.run(grads, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	    train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
	    if train_acc >= float(args.pru_acc):
	        counter = counter + 1
		accurancy = accurancy + train_acc	
            if counter == int(args.pru_v):
	        print("Retraining step %d, accurancy %g" % (j, train_acc))
		print("[Info] Training accurancy has became stable with Accurancy = %g" % (accurancy/counter))
		file_object.write("[Info] Pruning Retraining accurancy has became stable with Accurancy = %g \n" % (accurancy/counter))
		break
	    if (j % 1000 == 0):
	        train_acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		print("Retraining step %d, accurancy %g" % (j, train_acc))
	    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	array_wfc1 = sess.run(dense_w["w_fc1"])
	array_wfc2 = sess.run(dense_w["w_fc2"])

	print("none-zero in fc1 after retrain:%d / %d" % (np.sum(array_wfc1 != 0),np.size(array_wfc1)))
	print("none-zero in fc2 after retrain:%d / %d" % (np.sum(array_wfc2 != 0),np.size(array_wfc2)))
	te = time.clock()
	print("\n $$$$$$$$$$$$$ time checking3 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")

	t_pr_1 = time.clock()
	print("[Info] Prune Phase Time :\n Pruning = " + str((t_pr_1 - t_pr_0)*1000) + " ms \n")
	start = time.clock()
	test_acc = test(logit)
	end = time.clock()

	print("\033[0;31m Retraining: Iteration = %d , Test accurancy after pruning and retraining = %g , Inference time = %g ms \033[0m" % (i, test_acc,(end - start)*1000))
	print("\033[0;31m [Info]--------------Retraining End--------------------[Report]\033[0m")
	file_object.write("[Info] Prune Phase Time :\n Pruning = " + str((t_pr_1 - t_pr_0)*1000) + " ms \n")
	file_object.write("Retraining: Iteration = %d , Test accurancy after pruning and Retraining = %g , Inference time = %g ms\n" % (i, test_acc,(end - start)*1000))

	saver = tf.train.Saver(dense_w)
	saver.save(sess, "./models/model_ckpt_pruned_retrained", global_step=i)

    saver = tf.train.Saver(dense_w)
    saver.save(sess, "./models/model_ckpt_pruned_retrained")
    print("[Info] Successfully saved model_ckpt_pruned_retrained !")

# The Third Phase for Recovering
counter = 0
accurancy = 0
if(args.recover == True):
    print("\033[0;31m [Info]---------------------Recovering Start-------------------[Report]\033[0m")
    file_object.write("\n [Info]-------------------------Recovering Result List--------------------[Report]\n")
    t_rt_0 = time.clock()
    th = time.clock()
    saver = tf.train.Saver(dense_w)
    saver.restore(sess, args.sparse)
    te = time.clock()
    print("\n $$$$$$$$$$$$$ time checking0 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")
    th = time.clock()
    p = 1.0
    file_object.flush()
    th_fc1 = get_th(dense_w["w_fc1"], percentage=(1.0 - p))
    th_fc2 = get_th(dense_w["w_fc2"], percentage=(1.0 - p))
    sparse_w_fc1, idx_fc1 = prune(dense_w["w_fc1"], th_fc1)
    sparse_w_fc2, idx_fc2 = prune(dense_w["w_fc2"], th_fc2)
    sess.run(tf.assign(dense_w["w_fc1"], sparse_w_fc1))
    sess.run(tf.assign(dense_w["w_fc2"], sparse_w_fc2))

    array_w_fc1 = sess.run(dense_w["w_fc1"])
    array_w_fc2 = sess.run(dense_w["w_fc2"])

    print("none-zero in fc1 :%d / %d" % (np.sum(array_w_fc1 != 0),np.size(array_w_fc1)))
    print("none-zero in fc2 :%d / %d" % (np.sum(array_w_fc2 != 0),np.size(array_w_fc2)))

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        initial = tf.all_variables()
    else:
        initial = tf.global_variables()
    for var in initial:
        if sess.run(tf.is_variable_initialized(var)) == False:
            sess.run(var.initializer)

    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    logit = dense_cnn_model(x, dense_w, keep_prob)
       
    start = time.clock()
    test_acc = test(logit)
    end = time.clock()
    te = time.clock()
    print("\n $$$$$$$$$$$$$ time checking1 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")

    print("\033[0;31m Recovering : Test accurancy after recovering = %g , Inference time = %g ms \033[0m" % (test_acc,(end - start)*1000))
    print("\033[0;31m [Info]--------------Recovering End--------------------[Report]\033[0m")
    file_object.write("Recovering : Test accurancy after recovering = %g , Inference time = %g ms\n" % (test_acc,(end - start)*1000))
    th = time.clock()
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_)
    trainer = tf.train.AdamOptimizer(1e-4)
    grads = trainer.compute_gradients(cross_entropy)
    delete_none_grads(grads)

    count = 0
    for grad, var in grads:
        if (var.name == "w_fc1:0"):
            idx_in1 = tf.cast(tf.constant(idx_fc1), tf.float32)
            # grads[count] = (tf.multiply(idx_in1, grad), var)
            grads[count] = (tf.mul(idx_in1, grad), var)
        if (var.name == "w_fc2:0"):
            idx_in2 = tf.cast(tf.constant(idx_fc2), tf.float32)
            # grads[count] = (tf.multiply(idx_in2, grad), var)
            grads[count] = (tf.mul(idx_in2, grad), var)
        count += 1
    train_step = trainer.apply_gradients(grads)

    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        initial = tf.all_variables()
    else:
        initial = tf.global_variables()
    for var in initial:
        if sess.run(tf.is_variable_initialized(var)) == False:
            sess.run(var.initializer)
    te = time.clock()
    print("\n $$$$$$$$$$$$$ time checking2 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")
    print("\033[0;31m [Info]--------------Tuning Start--------------------[Report]\033[0m")
    th = time.clock()
    for j in range(int(args.train_iterations)):
        batch = mnist.train.next_batch(50)
        idx_in1_value = sess.run(idx_in1)
        grads_fc1_value = sess.run(grads, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if (j % 200 == 0):
            train_acc = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print("Tuning step %d, accurancy %g" % (j, train_acc))
        
        train_acc = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        if train_acc > float(args.rec_acc):
            counter = counter + 1
            accurancy = accurancy + train_acc
        if counter == int(args.rec_v):
             print("Tuning step %d, accurancy %g" % (j,train_acc))
             print("[Info] Training accurancy has became stable with Accurancy = %g" % (accurancy/counter))
             file_object.write("[Info] Recovering and tuning accurancy has became stable with Accurancy = %g \n" % (accurancy/counter))
             break

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    array_wfc1 = sess.run(dense_w["w_fc1"])
    array_wfc2 = sess.run(dense_w["w_fc2"])

    print("none-zero in fc1 after tuning:%d / %d" % (np.sum(array_wfc1 != 0),np.size(array_wfc1)))
    print("none-zero in fc2 after tuning:%d / %d" % (np.sum(array_wfc2 != 0),np.size(array_wfc2)))
    te = time.clock()
    print("\n $$$$$$$$$$$$$ time checking3 = " + str((te - th)*1000)+"ms $$$$$$$$$$$$$$$ \n")
    t_rt_1 = time.clock()
    print("[Info] Recover Phase Time :\n Recovering = " + str((t_rt_1 - t_rt_0)*1000) + " ms \n")
    start = time.clock()
    test_acc = test(logit)
    end = time.clock()

    print("\033[0;31m Tuning: Test accurancy after recovering and tuning = %g , Inference time = %g ms \033[0m" % (test_acc,(end - start)*1000))
    print("\033[0;31m [Info]--------------Tuning End--------------------[Report]\033[0m")
    file_object.write("[Info] Recover Phase Time :\n Recovering = " + str((t_rt_1 - t_rt_0)*1000) + " ms \n")
    file_object.write("Recovering: Test accurancy after recovering and tuning = %g , Inference time = %g ms\n" % (test_acc,(end - start)*1000))

    saver = tf.train.Saver(dense_w)
    saver.save(sess, "./models/model_ckpt_recover_tuned")
    print("[Info] Successfully saved model_ckpt_recover_tuned !")
    #print("[Info] Time total:" + str((t_pt_1 - t_pt_0)*1000 + (t_pr_1 - t_pr_0)*1000 + (t_rt_1 - t_rt_0)*1000) + " ms \n")
file_object.close()
