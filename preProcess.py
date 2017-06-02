from gensim.models import Doc2Vec
import codecs
import csv

inputFile = open("./data/movieData.csv")
outputFile = open("./data/label.data", "w")

# Key value
mapCat = {'Adventure':0, 'Animation':1, 'Children':2, 'Comedy':3, 'Fantasy':4, 'Romance':5, 'Drama':6,
          'Action':7, 'Crime':8, 'Thriller':9, 'Mystery':10, 'Horror':11, 'Sci-Fi':12, 'Documentary':13, 'IMAX':14,
          'War':15, 'Musical':16, 'Western':17, 'Film-Noir':18, '(no genres listed)':19}

# Vector model
model = "./enwiki_dbow/doc2vec.bin"
startAlpha = 0.01
inferEpoch = 1000
model = Doc2Vec.load(model)

for line in csv.reader(inputFile, skipinitialspace=True):
    # Skip some ID
    if len(line) != 4:
        continue

    print line[0]

    # For label data
    outputFile.write(line[0] + ",\"" + line[1] + "\"," + "|".join([str(mapCat[x]) for x in line[2].split('|')]) + "," + str(len(line[3].split(".")[:-1])) + "\n")

    # For text in each movie
    with open("./data/" + line[0] + ".text", "w") as text:
        for sentence in line[3].split(".")[:-1]:
            text.write(sentence.lstrip() + ".\n")

    # Sentence vector for each movie
    testDocs = [x.strip().split() for x in codecs.open("./data/" + line[0] + ".text", "r", "utf-8").readlines()]
    with open("./data/" + line[0] + ".vec", "w") as vec:
        for d in testDocs:
            vec.write(" ".join([str(x) for x in model.infer_vector(d, alpha=startAlpha, steps=inferEpoch)]) + "\n")
