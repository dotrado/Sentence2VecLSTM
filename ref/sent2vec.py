import logging
import sys
import gensim
import codecs
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models import Doc2Vec

# # Training
# logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
# logging.info("running %s" % " ".join(sys.argv))
# inputFile = 'test.txt'
# model = Doc2Vec(TaggedLineDocument(inputFile), size=100, window=8, min_count=5, workers=4)
# model.save('model.model')
# model1 = Doc2Vec.load('model.model')
# print model1.docvecs[0][0]
# print model1.infer_vector("Harbin Institute of Technology (HIT) was founded in 1920.".split())[0]

# Inferences
model= "./enwiki_dbow/doc2vec.bin"
testDocs= "./test.txt"
outputFile= "./testVectors.txt"

# Inference hyper-parameters
startAlpha = 0.01
inferEpoch = 1000

#load model
model = Doc2Vec.load(model)
testDocs = [x.strip().split() for x in codecs.open(testDocs, "r", "utf-8").readlines()]

#infer test vectors
output = open(outputFile, "w")
for d in testDocs:
    output.write(" ".join([str(x) for x in model.infer_vector(d, alpha=startAlpha, steps=inferEpoch)]) + "\n" )
output.flush()
output.close()
