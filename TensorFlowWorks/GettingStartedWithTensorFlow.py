from __future__ import print_function
import tensorflow as tf
import numpy as np


sess = tf.Session()

def step1():
    node1 = tf.constant("rahat")
    node2 = tf.constant(" ibn rafiq")
    node3 = tf.add(node1, node2, "node3")
    print(node3)
    
    
def step2():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)
    add_and_triple = adder_node * 3.
    print(sess.run(add_and_triple, {a: [1, 3, 0], b: [2, 4, 10]}))


def step3():
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W*x + b
    init = tf.global_variables_initializer()
    sess.run(init)
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


def step4():
    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    
    
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W*x + b
    y = tf.placeholder(tf.float32)
    
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    
    
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    
    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    
    
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})
    
    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    


def step5():
    feature_columns = [tf.feature_column.numeric_column("x",shape=[1])]
    
    
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    
    input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=40,num_epochs=None,shuffle=True)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_train},y_train,batch_size=40,num_epochs=1000,shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_eval},y_eval,batch_size=40,num_epochs=1000,shuffle=False)
    
    estimator.train(input_fn=input_fn,steps=1000)
    
    
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    
    print("train metrics: %r"% train_metrics)
    print("eval metrics: %r"% eval_metrics)
    
    
    
    
       
    
step5()


