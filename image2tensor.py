import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split

# Load image resize to size (100,100), grayscale, and whiten  
def imageload(file):
	pic=Image.open(file)
	pic=pic.resize((200,200)).convert('L')
	pic_array=np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 1)
	pic_array_white= pic_array/255.
	return pic_array_white

def convert2tensor(array):
	tensor = tf.pack(array)
        return tensor

path = "/home/matthew/Desktop/train"
files = []
labels = []
for i in ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]:
	files = files + ["%s/%s/%s" %(path,i,f) for f in listdir("%s/%s" %(path,i)) if isfile(join("%s/%s" %(path,i), f))]
	labels= labels + [i for p in listdir("%s/%s" % (path, i)) if isfile(join("%s/%s" % (path, i), p))]
print(files[0:3])
print(labels[0:3])

#Splits files and labels into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(files, labels)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test)

image_array=[]

# converts files to array not tensor
def files2tensor(files):
	for file in files:
		image_array.append(imageload(file))
	return image_array



def one_hot_encoding(array):
	one=pd.get_dummies(array)

	one_hot=one[["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]].values.tolist()

	return one_hot
# Import data into array

traintensordata = files2tensor(X_train)
traintensorlabel = one_hot_encoding(y_train)
testtensordata = files2tensor(X_test)
testtensorlabel = one_hot_encoding(y_test)
valtensordata = files2tensor(X_val)
valtensorlabel = one_hot_encoding(y_val) 

#print(traintensordata)


# Parameters
learning_rate = 0.01
training_iters = 200000
batch_size = 10
display_step = 10

# Network Parameters
n_input = 40000 # Photo data input (img shape: 100*100)
n_classes = 8 # Total Classes
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 200,200,1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=5)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 5x5 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([5, 5, 128, 64])),
    # fully connected, 5*5*128 inputs, 2048 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*64, 2048])),
    # 2048 inputs, 8 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([2048, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([2048])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #Run for 1000 Epochs
    for i in range(0,200):
    	step = 1
    	# Keep training until reach max iterations (modification until I can get this to work correctly)
    	while step * batch_size < len(X_train):
        	batch_x= traintensordata[(step-1)*batch_size % len(X_train):step*batch_size % len(X_train)]
        	batch_y= traintensorlabel[(step-1)*batch_size % len(X_train):step*batch_size % len(X_train)]
        	# Run optimization op (backprop)
        	sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        	if step % display_step == 0:
            	# Calculate validation loss and accuracy
            		loss, acc = sess.run([cost, accuracy], feed_dict={x: valtensordata[:100], y: valtensorlabel[:100], keep_prob: 1.})
            		print("Epoch "+ str(i)+", Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                	 	 "{:.6f}".format(loss) + ", Training Accuracy= " + \
                 	  	"{:.5f}".format(acc))
        	step += 1
    loss, acc = sess.run([cost, accuracy], feed_dict={x: testtensordata[:200], y: testtensorlabel[:200], keep_prob: 1.})
    print("Testing Accuracy: "+ "{:.5f}".format(acc)+", Cost: "+"{:.6f}".format(loss))
    print("Optimization Finished!")

   
