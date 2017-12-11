from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Tweaked the architecture  that was in the tensowflow website. added Adam Optimizer and another two
fully connected layers with 2048 and 1024 neurons respectively. dropout is 0.5 instead of 0.4.
step size is reduced to 1000 and the final accuracy was 99.1 percent. The tensorflow website's one
had 20000 steps and had an accuracy of 97.3 percent. 
link:https://www.tensorflow.org/tutorials/layers


"""

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    
    input_layer = tf.reshape(features["x"], [-1,28,28,1])
    conv1 = tf.layers.conv2d(inputs=input_layer,
                            filters=32,
                            kernel_size = [5,5],
                            padding="same",
                            activation=tf.nn.relu
                           )
    pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                    pool_size = [2,2], 
                                    strides = 2)
    
    
    conv2 = tf.layers.conv2d(inputs = pool1, 
                             filters = 64, 
                             kernel_size = [5,5], 
                             padding = "same", 
                             activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
    
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
    
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout2, units=10)
    
    predictions = {
                   "classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels,logits = logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        
    


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                     x={"x": train_data},
                     y=train_labels,
                     batch_size=100,
                     num_epochs=None,
                     shuffle=True)
    mnist_classifier.train(
                           input_fn=train_input_fn,
                           steps=1000)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()