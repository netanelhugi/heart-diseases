import tensorflow as tf
import numpy as np
import random

# Cancels the warning message of tensorFlow:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For print the table:
# from texttable import Texttable


# Count the number of lines.
fileForCount = open("DataSet/processed.cleveland.data")
num_lines = sum(1 for line in fileForCount)

# Open the file and read the data.
file = open("DataSet/processed.cleveland.data")
rows = file.readlines()

# Split the data between test and train:
# 70% data for train
trainData = []
trainResults = []
# 30% data for test
testData = []
testResults = []

# shuffle the data, for divide the data randomly.
random.shuffle(rows)

count = 0
for i in rows:
    count = count + 1
    words = i.split(",")  # Split the sentence into words.
    words[-1] = words[-1].strip()  # remove '\n' from the last word.

    # convert to np.array.
    sentence = np.array(words)
    patient = sentence.astype(np.float32)

    # if the value in the last index is > 0, the patient has a presence of heart disease:
    # change the value to '1'.
    if (patient[-1] > 0):
        patient[-1] = 1

    # A division between test and train.
    if ((count / num_lines) < 0.7):
        result = []
        result.append(patient[-1])
        trainResults.append(result)
        trainData.append(patient[:13])
    else:
        result = []
        result.append(patient[-1])
        testResults.append(result)
        testData.append(patient[:13])


# The logistic function.
def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


features = 13
eps = 1e-12
alpah = 0.0001
iterations = 100000

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features, 1]))
b = tf.Variable(tf.zeros([1]))

# Model function.
y = 1 / (1.0 + tf.exp(-(tf.matmul(x, W) + b)))

# Loss function
loss1 = -(y_ * tf.log(y + eps) + (1 - y_) * tf.log(1 - y + eps))
loss = tf.reduce_mean(loss1)

update = tf.train.GradientDescentOptimizer(alpah).minimize(loss)

# Graph parameters.
# m = tf.Variable(0.)
# msum = tf.summary.scalar('m', m)
# losssum = tf.summary.scalar('loss', loss)
# merged = tf.summary.merge_all()

sess = tf.Session()
# file_writer = tf.summary.FileWriter('./my_graph', sess.graph)
sess.run(tf.global_variables_initializer())

for j in range(0, iterations):
    # for graph:
    # [_, curr_sammary] = sess.run([update, merged], feed_dict={x: trainData, y_: trainResults})
    # file_writer.add_summary(curr_sammary, j)

    sess.run(update, feed_dict={x: trainData, y_: trainResults})
    if j % 10000 == 0:
        # print('Iteration:', j, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:',
        #       loss.eval(session=sess, feed_dict={x: trainData, y_: trainResults}))
        print('Iteration:', j, ' loss:',loss.eval(session=sess, feed_dict={x: trainData, y_: trainResults}))

# Check train accuracy.
true = 0

# test every patient in test data.
for i in range(len(trainData)):
    person = trainData[i]
    ans = trainResults[i]

    # Real value: 0-Absence/1-Presence of heart disease.
    real = int(ans[0])
    # Predicted Value:
    pred = logistic_fun(np.matmul(person, sess.run(W)) + sess.run(b))
    predVal = int(np.round(pred))

    if (predVal == real):
        true += 1

print(" ")
TrainAccuracy = true / len(trainData)
print("Train accuracy:  %.3f" % TrainAccuracy)



# Testing the model:
true = 0
truePositive = 0
falsePositive = 0
trueNegative = 0
falseNegative = 0

# test every patient in test data.
for i in range(len(testData)):
    person = testData[i]
    ans = testResults[i]

    # Real value: 0-Absence/1-Presence of heart disease.
    real = int(ans[0])
    # Predicted Value:
    pred = logistic_fun(np.matmul(person, sess.run(W)) + sess.run(b))
    predVal = int(np.round(pred))

    if (predVal == real):
        true += 1
    if (real == 1):
        if (predVal == 1):
            truePositive += 1
        else:
            falseNegative += 1
    else:
        if (predVal == 1):
            falsePositive += 1
        else:
            trueNegative += 1

# print the summary of results:
print(" ")
print("Alpah: ", alpah)
print("Iterations: ", iterations)

print(" ")

print("Total patients: ", len(trainData) + len(testData))
print("Train: ", len(trainData))
print("Test: ", len(testData))

print(" ")
print("Results: %d / %d" % (true, len(testData)))

# # Table print
# t = Texttable()
# t.add_rows([[' ', 'Classified as Positive:%d' % (truePositive + falsePositive),
#              'Classified as Negative:%d' % (trueNegative + falseNegative)],
#             ['Really Positive:%d' % (truePositive + falseNegative), 'True Positive: %d' % truePositive,
#              'False Negative: %d' % falseNegative],
#             ['Really Negative:%d' % (falsePositive + trueNegative), 'False Positive: %d' % falsePositive,
#              'True Negative: %d' % trueNegative]])
# print(t.draw())

print(" ")

accuracy = true / len(testData)
print("Accuracy:  %.3f" % accuracy)
recall = truePositive / (truePositive + falseNegative)
print("Recall:    %.3f" % recall)
precision = truePositive / (truePositive + falsePositive)
print("Precision: %.3f" % precision)
fMeasure = 2 * (precision * recall) / (precision + recall)
print("F-Measure: %.3f" % fMeasure)

# For graph:
# file_writer.close()
