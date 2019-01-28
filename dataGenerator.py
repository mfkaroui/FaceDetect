import numpy as np
import tensorflow as tf


'''
The data generator will be responsible for subdividing the input data into training sets and test sets
It will subsample classes per batch to solve class imbalance issues.

'''
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batchSize=100, validationSplit=0.5):
        nclasses = labels.shape[1]
        classCounts = np.zeros((nclasses,), dtype=int)
        for i in range(nclasses):
            classCounts[i] = (labels[labels[:,i] == 1]).shape[0]
        self.maxSampleSize = min(np.min(classCounts), int(batchSize / nclasses))
        self.batchSize = self.maxSampleSize * nclasses
        

        