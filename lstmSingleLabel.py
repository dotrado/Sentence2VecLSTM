import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import numpy as np
import tensorflow as tf

path = "./data/20newDataset/"

inputFile = open(path + "label.data")

# Keep all input that has different size of T (Number of sentences)
# For each key [batch, sizeT, sizeVec]
xDict = dict()
# For each key [batch, sizeClass]
yDict = dict()

# Read all input data
for line in csv.reader(inputFile, skipinitialspace=True):
    tmp = []
    if line[3] not in xDict:
        xDict[line[3]] = []
        yDict[line[3]] = []
    with open(path + line[0] + ".vec", "r") as vec:
        countSen = 0
        for sentence in vec:
            if countSen < int(line[3]):
                tmp.append([float(i) for i in sentence.split()])
                countSen += 1
        xDict[line[3]].append(tmp)

    yTmp = [0] * 20
    for i in line[2].split("|"):
        yTmp[int(i)] = 1
    yDict[line[3]].append(yTmp)


# Convert to numpy array
for key, value in xDict.iteritems():
    xDict[key] = np.array(value)
    yDict[key] = np.array(yDict[key])


# Split Data
xDictTrain = dict()
yDictTrain = dict()
xDictTest = dict()
yDictTest = dict()
for key, value in xDict.iteritems():
    randSet = np.random.rand(value.shape[0]) < 0.8
    xDictTrain[key] = xDict[key][randSet]
    yDictTrain[key] = yDict[key][randSet]
    xDictTest[key] = xDict[key][~randSet]
    yDictTest[key] = yDict[key][~randSet]


# Check the dimension
for key, value in xDict.iteritems():
    print (key, xDictTrain[key].shape, yDictTrain[key].shape, xDictTest[key].shape, yDictTest[key].shape)


# print xDict["3"][0]
# print yDict["3"][0]


lstmSize = 200
outputSize = 20
batchSize = 100
layerSize = 2
stepSize = 8

# Placeholder for input and output
x = tf.placeholder(tf.float32, [None, None, 300])
y = tf.placeholder(tf.float32, [None, 20])
keepProb = tf.placeholder(tf.float32)

# Build network
lstm = tf.contrib.rnn.BasicLSTMCell(lstmSize, forget_bias=1.0, state_is_tuple=True)
lstmDrop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keepProb)
cell = tf.contrib.rnn.MultiRNNCell([lstmDrop] * layerSize, state_is_tuple=True)
initialState = cell.zero_state(batchSize, tf.float32)

# # Calculate each step cell at time t
# outputs = []
# state = initialState
# for timeStep in range(stepSize):
#     if timeStep > 0: tf.get_variable_scope().reuse_variables()
#     cellOutput, state = cell(x[:, timeStep, :], state)
#     outputs.append(cellOutput)

# Calculate each step cell at time t
outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=initialState)
outputs = outputs[:, -1, :]

# Affine layer to make output to 20 classes
W = tf.Variable(tf.random_normal([lstmSize, outputSize]), trainable=True)
B = tf.Variable(tf.random_normal([outputSize]))

# Predict
predict = tf.matmul(outputs, W) + B

# Define loss
LabelPredict = tf.argmax(predict, 1)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict)
meanLoss = tf.reduce_mean(loss)
trainStep = tf.train.GradientDescentOptimizer(1e-2).minimize(meanLoss)

# Define accuracy
correctPred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Initial session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# For Tensorboard
averageLoss = tf.placeholder(tf.float32)
averageSummary = tf.summary.scalar("averageLoss", averageLoss)
path = "./logs/lstm"
merge = tf.summary.merge_all()
if tf.gfile.Exists(path):
    tf.gfile.DeleteRecursively(path)
    tf.gfile.MkDir(path)
else:
    tf.gfile.MakeDirs(path)
write = tf.summary.FileWriter(path, sess.graph)

# Run session
for epoch in range(500):
    avgLoss = []
    averageAccuracy = []
    for key, value in xDictTrain.iteritems():
        if key == "0":
            continue
        if xDictTrain[key].shape[0] > batchSize:
            for i in range(0, xDictTrain[key].shape[0], batchSize):
                xBatch = xDictTrain[key][i:i + batchSize]
                yBatch = yDictTrain[key][i:i + batchSize]
                if xBatch.shape[0] != batchSize:
                    perm = np.arange(xDictTrain[key].shape[0])
                    np.random.shuffle(perm)
                    xDictTrain[key] = xDictTrain[key][perm]
                    yDictTrain[key] = yDictTrain[key][perm]
                    xBatch = xDictTrain[key][0:batchSize]
                    yBatch = yDictTrain[key][0:batchSize]
                _, loss, acc = sess.run([trainStep, meanLoss, accuracy], feed_dict={x:xBatch, y:yBatch, keepProb:0.8})
                avgLoss.append(loss)
                averageAccuracy.append(acc)
    summary = sess.run(merge, feed_dict={averageLoss: np.mean(avgLoss)})
    write.add_summary(summary, epoch)

    # Accuracy Test
    averageAccuracyTest = []
    for key, value in xDictTest.iteritems():
        if key == "0":
            continue
        if xDictTest[key].shape[0] > batchSize:
            for i in range(0, xDictTest[key].shape[0], batchSize):
                xBatch = xDictTest[key][i:i + batchSize]
                yBatch = yDictTest[key][i:i + batchSize]
                if xBatch.shape[0] != batchSize:
                    perm = np.arange(xDictTest[key].shape[0])
                    np.random.shuffle(perm)
                    xDictTest[key] = xDictTest[key][perm]
                    yDictTest[key] = yDictTest[key][perm]
                    xBatch = xDictTest[key][0:batchSize]
                    yBatch = yDictTest[key][0:batchSize]
                acc = sess.run(accuracy, feed_dict={x: xBatch, y: yBatch, keepProb:1.0})
                averageAccuracyTest.append(acc)

    print (epoch, np.mean(avgLoss), np.mean(averageAccuracy), np.mean(averageAccuracyTest))

# # Checking
# print sess.run(LabelPredict, feed_dict={x:xDict["8"][:batchSize]})[0]
# print yDict["8"][0]

# # Accuracy Test
# averageAccuracyTest = []
# for key, value in xDictTest.iteritems():
#     if key == "0":
#         continue
#     if xDictTest[key].shape[0] > batchSize:
#         for i in range(0, xDictTest[key].shape[0], batchSize):
#             xBatch = xDictTest[key][i:i + batchSize]
#             yBatch = yDictTest[key][i:i + batchSize]
#             if xBatch.shape[0] != batchSize:
#                 perm = np.arange(xDictTest[key].shape[0])
#                 np.random.shuffle(perm)
#                 xDictTest[key] = xDictTest[key][perm]
#                 yDictTest[key] = yDictTest[key][perm]
#                 xBatch = xDictTest[key][0:batchSize]
#                 yBatch = yDictTest[key][0:batchSize]
#             acc = sess.run(accuracy, feed_dict={x:xBatch, y:yBatch})
#             averageAccuracyTest.append(acc)
# print np.mean(averageAccuracyTest)