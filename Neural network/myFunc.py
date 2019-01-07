import tensorflow as tf
import numpy as np
import random

# The logistic function.
def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))

def results(data,results,sess,W,b):
    # Testing the model:
    true = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    # test every patient in data.
    for i in range(len(data)):
        person = data[i]
        ans = results[i]

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


    accuracy = true / len(data)
    recall = truePositive / (truePositive + falseNegative)
    precision = 0
    fMeasure = 0

    if(truePositive + falsePositive>0):
        precision = truePositive / (truePositive + falsePositive)
    if(precision + recall>0):
        fMeasure = 2 * (precision * recall) / (precision + recall)

    return accuracy,recall,precision,fMeasure,true


def splitData(file_dir):
    # Count the number of lines.
    fileForCount = open(file_dir)
    num_lines = sum(1 for line in fileForCount)

    # Open the file and read the data.
    file = open(file_dir)
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

    avg = ['56', '1', '3', '134', '251', '0', '1', '139', '1', '1', '2', '1', '5']

    count = 0
    for i in rows:
        count = count + 1
        words = i.split(",")  # Split the sentence into words.
        words[-1] = words[-1].strip()  # remove '\n' from the last word.

        for j in range(0, 13):

            if (words[j] == '?'):
                words[j] = avg[j]

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

    return trainData,trainResults,testData,testResults


def resultsMLP(data,results,sess,z2,x):
    # Testing the model:
    true = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    y = tf.nn.sigmoid(z2)

    pred = y.eval(session=sess, feed_dict={x: data})

    # test every patient in data.
    for i in range(len(data)):
        person = data[i]
        ans = results[i]

        # Real value: 0-Absence/1-Presence of heart disease.
        real = int(ans[0])
        # Predicted Value:
        # print(pred[i])
        predVal = int(np.round(pred[i]))

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


    accuracy = true / len(data)
    recall = truePositive / (truePositive + falseNegative)
    precision = 0
    fMeasure = 0

    if(truePositive + falsePositive>0):
        precision = truePositive / (truePositive + falsePositive)
    if(precision + recall>0):
        fMeasure = 2 * (precision * recall) / (precision + recall)

    return accuracy,recall,precision,fMeasure,true



