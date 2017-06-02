import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import numpy as np
import tensorflow as tf


inputFile = open("./data/label.data")

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
    with open("./data/" + line[0] + ".vec", "r") as vec:
      for sentence in vec:
            tmp.append([float(i) for i in sentence.split()])
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


print xDict["8"][0]
print yDict["8"][0]


lstmSize = 200
outputSize = 20
batchSize = 100
layerSize = 2
stepSize = 8

# Placeholder for input and output
x = tf.placeholder(tf.float32, [None, None, 300])
y = tf.placeholder(tf.float32, [None, 20])

# Build network
lstm = tf.contrib.rnn.BasicLSTMCell(lstmSize, forget_bias=1.0, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([lstm] * layerSize, state_is_tuple=True)
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

# Define loss
predict = tf.matmul(outputs, W) + B
sigmoidPredict = tf.sigmoid(predict)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict)
meanLoss = tf.reduce_mean(loss)
trainStep = tf.train.GradientDescentOptimizer(1e-3).minimize(meanLoss)

# Define accuracy
predictLabel = tf.cast(tf.greater(predict, 0.5), tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predictLabel), tf.float32))

# All label correct
allLabelsTrue = tf.reduce_min(tf.cast(tf.equal(y, predictLabel), tf.float32), 1)
accuracy2 = tf.reduce_mean(allLabelsTrue)

# Intersection over union
intersect = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(predict, tf.bool), tf.cast(y, tf.bool)), tf.float32), 1)
union = tf.reduce_sum(tf.cast(tf.logical_or(tf.cast(predict, tf.bool), tf.cast(y, tf.bool)), tf.float32), 1)
accuracy3 = tf.reduce_mean(tf.divide(intersect, tf.add(union, 1e-4)))

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
for epoch in range(1000):
    avgLoss = []
    averageAccuracy = []
    averageAccuracy2 = []
    averageAccuracy3 = []
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
                _, loss, acc1, acc2, acc3 = sess.run([trainStep, meanLoss, accuracy, accuracy2, accuracy3], feed_dict={x:xBatch, y:yBatch})
                avgLoss.append(loss)
                averageAccuracy.append(acc1)
                averageAccuracy2.append(acc2)
                averageAccuracy3.append(acc3)
    summary = sess.run(merge, feed_dict={averageLoss: np.mean(avgLoss)})
    write.add_summary(summary, epoch)
    print (epoch, np.mean(avgLoss), np.mean(averageAccuracy), np.mean(averageAccuracy2), np.mean(averageAccuracy3))

# Checking
print sess.run(sigmoidPredict, feed_dict={x:xDict["8"][:batchSize]})[0]
print yDict["8"][0]

# Accuracy
averageAccuracy = []
averageAccuracy2 = []
averageAccuracy3 = []
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
            acc1, acc2, acc3 = sess.run([accuracy, accuracy2, accuracy3], feed_dict={x:xBatch, y:yBatch})
            averageAccuracy.append(acc1)
            averageAccuracy2.append(acc2)
            averageAccuracy3.append(acc3)
print (np.mean(averageAccuracy), np.mean(averageAccuracy2), np.mean(averageAccuracy3))