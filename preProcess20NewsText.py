from gensim.models import Doc2Vec
import codecs
import csv
import re
from sklearn.datasets import fetch_20newsgroups

outputFile = open("./data/20newDataset/label.data", "w")

# Get 20 news data set
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Check dimension
print (newsgroups_test.target.shape[0], newsgroups_train.target.shape[0])

# Vector model
model = "./enwiki_dbow/doc2vec.bin"
startAlpha = 0.01
inferEpoch = 1000
model = Doc2Vec.load(model)

# Training data
for line in range(newsgroups_train.target.shape[0]):
    print line

    # Read text
    textData = re.sub("[,]|\n+"," ",newsgroups_train.data[line])

    # For label data
    outputFile.write(str(line) + ",\"" + " " + "\"," + str(newsgroups_train.target[line]) + "," + str(len(textData.split(".")[:-1])) + "\n")

    # For text in each movie
    with open("./data/20newDataset/" + str(line) + ".text", "w") as text:
        for sentence in textData.split(".")[:-1]:
            text.write(sentence.lstrip().encode("utf-8") + ".\n")

    # Sentence vector for each movie
    testDocs = [x.strip().split() for x in codecs.open("./data/20newDataset/" + str(line) + ".text", "r", "utf-8").readlines()]
    with open("./data/20newDataset/" + str(line) + ".vec", "w") as vec:
        for d in testDocs:
            vec.write(" ".join([str(x) for x in model.infer_vector(d, alpha=startAlpha, steps=inferEpoch)]) + "\n")

# Testing data
for index in range(newsgroups_test.target.shape[0]):
    line = index + newsgroups_train.target.shape[0]
    print line

    # Read text
    textDate = re.sub("[,]|\n+"," ",newsgroups_test.data[index])

    # For label data
    outputFile.write(str(line) + ",\"" + " " + "\"," + str(newsgroups_test.target[index]) + "," + str(len(textDate.split(".")[:-1])) + "\n")

    # For text in each movie
    with open("./data/20newDataset/" + str(line) + ".text", "w") as text:
        for sentence in textDate.split(".")[:-1]:
            text.write(sentence.lstrip().encode("utf-8") + ".\n")

    # Sentence vector for each movie
    testDocs = [x.strip().split() for x in codecs.open("./data/20newDataset/" + str(line) + ".text", "r", "utf-8").readlines()]
    with open("./data/20newDataset/" + str(line) + ".vec", "w") as vec:
        for d in testDocs:
            vec.write(" ".join([str(x) for x in model.infer_vector(d, alpha=startAlpha, steps=inferEpoch)]) + "\n")