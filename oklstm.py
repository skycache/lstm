'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import numpy as np
import tensorflow as tf
import data_helpers
import cPickle

# Import MINST data

'''
To classify images using a reccurent neural network, we consider every image
row as a sequence of pixels. 
Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

"""
hyper parameters 
"""
# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 10
display_step = 5

# Network Parameters
n_input = 300 # MNIST data input (img shape: 28*28)
n_steps = 340 # timesteps

n_hidden = 128 # hidden layer num of features, output dim
n_classes = 2 # MNIST total classes (0-9 digits)


"""
data
"""
print("Loading data...")
x = cPickle.load(open("/home/zjj/distant_s/data/mrless", "rb"))
x, y = x[0], x[1]
print("Loading done")
print x.shape
print y.shape

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# TODO: This is very crude, should use cross-validation
val_posi = -1 * int(len(x_shuffled) * 0.1)
x_train, x_dev = x_shuffled[:val_posi], x_shuffled[val_posi:]
y_train, y_dev = y_shuffled[:val_posi], y_shuffled[val_posi:]

"""
data batch
"""



"""
tensorflow define symbol
"""
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_steps * n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

"""
one single train step
"""
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_hidden)

    """
    (batch_size, step, n_input) to list, len: n_step of (batch_size, n_input)
    """
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

    # use all output
    outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden * n_steps])
    # [28, 128, 128] to [128, 128 * 28]
    # outputs = tf.slice(outputs, [], [n_hidden])
    softmax = tf.nn.xw_plus_b(outputs, weights['out'], biases['out'])

    # only use last output
    # Linear activation, using rnn inner loop last output
    # softmax output
    # softmax = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return softmax

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    batches = data_helpers.batch_iter(zip(x_train, y_train) ,batch_size, training_iters)
    # Keep training until reach max iterations
    for batch in batches:
        batch_x, batch_y = zip(*batch)
        # batch_x = np.array(batch_x).reshape((batch_size, n_steps, n_input))
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    test_data = x_dev
    test_label = y_dev
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
