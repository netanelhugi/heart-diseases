import tensorflow as tf
import myFunc as my

dataFile = "DataSet/processed.cleveland.data"

trainData,trainResults,testData,testResults = my.splitData(dataFile)

features = 13
hidden_layer_nodes1 = 10
alpah = 0.0001
iterations = 70000

# Input layer:
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])

# Hidden layer:
W1 = tf.Variable(tf.truncated_normal([features, hidden_layer_nodes1], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes1]))
z1 = tf.add(tf.matmul(x,W1),b1)

# Activtion layer:
a1 = tf.nn.relu(z1)

# Output layer:
W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes1,1], stddev=0.1))
b2 = tf.Variable(0.)
z2 = tf.matmul(a1,W2)+b2
y = tf.nn.sigmoid(z2)

loss = tf.reduce_mean(-(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y)))
update = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for j in range(0, iterations):
    sess.run(update, feed_dict={x: trainData, y_: trainResults})
    if j % 10000 == 0:
        print('Iteration:', j, ' loss:',loss.eval(session=sess, feed_dict={x: trainData, y_: trainResults}))

# train testing:
accuracyTrain, recallTrain, precisionTrain, fMeasureTrain,trueTrain = my.resultsMLP(trainData,trainResults,sess,z2,x)

print(" ")
print("Train accuracy:  %.3f" % accuracyTrain)

# Testing the model:
accuracyTest, recallTest, precisionTest, fMeasureTest,trueTest = my.resultsMLP(testData,testResults,sess,z2,x)

# print the summary of results:
print(" ")
print("Summary:")
print("Alpah: ", alpah)
print("Iterations: ", iterations)

print(" ")

print("Total patients: ", len(trainData) + len(testData))
print("Train: ", len(trainData))
print("Test: ", len(testData))

print(" ")
print("Test results: %d / %d" % (trueTest, len(testData)))

print(" ")
print("Accuracy:  %.3f" % accuracyTest)
print("Recall:    %.3f" % recallTest)
print("Precision: %.3f" % precisionTest)
print("F-Measure: %.3f" % fMeasureTest)




