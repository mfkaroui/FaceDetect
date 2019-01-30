import numpy as np
import tensorflow as tf
import random
from PIL import Image

'''
The data generator will be responsible for subdividing the input data into training sets and test sets
It will subsample classes per batch to solve class imbalance issues.
'''
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batchCount = 10):
        #store the batchCount
        self.batchCount = batchCount
        #get the number of classes from the onehotencoded labels
        self.nclasses = labels.shape[1]
        #initialize the class counts to zero
        self.classCounts = np.zeros((self.nclasses,), dtype=int)
        #initialize the data array
        self.data = []
        #iterate through each class
        for i in range(self.nclasses):
            #select data that matches the current class
            truthTable = labels[:,i] == 1
            #zip input and output data together
            self.data.append([images[truthTable], labels[truthTable]])
            #count the number of samples in the class
            self.classCounts[i] = self.data[-1][0].shape[0]
            #our intial shuffle
            indexes = random.sample(range(self.data[-1][0].shape[0]), self.data[-1][0].shape[0])
            self.data[-1][0] = self.data[-1][0][indexes]
            self.data[-1][1] = self.data[-1][1][indexes]
        #figure out how many samples to choose per epoch per class
        self.maxSampleSize = np.min(self.classCounts)
        self.batchSize = int(np.floor((self.maxSampleSize * self.nclasses) / self.batchCount))
        #sample the data
        self.sampleData()

    def sampleData(self):
        self.images = []
        self.labels = []
        #iterate through each class
        for i in range(self.nclasses):
            #select from each class randomly
            indexes = random.sample(range(self.data[i][0].shape[0]), self.maxSampleSize)
            self.images.extend(self.data[i][0][indexes])
            self.labels.extend(self.data[i][1][indexes])
        self.images = np.stack(self.images, axis=0)
        self.labels = np.stack(self.labels, axis=0)
        #final shuffle
        indexes = random.sample(range(self.maxSampleSize * self.nclasses), self.maxSampleSize * self.nclasses)
        self.images = self.images[indexes]
        self.labels = self.labels[indexes]

    def on_epoch_end(self):
        #when we finish an epoch we will generate a new batch
        self.sampleData()

    def __len__(self):
        return self.batchCount

    def __getitem__(self, index):
        s = slice(index * self.batchSize, (index + 1) * self.batchSize)
        images = []
        for imgPath in self.images[s]:
            img = Image.open(imgPath)
            images.append(np.asarray(img) / 255)
            img.close()
        return np.stack(images, axis=0), self.labels[s]

    def split(self, ratio):
        images1 = []
        images2 = []

        labels1 = []
        labels2 = []
        for i in range(self.nclasses):
            splitIndex = int(self.classCounts[i] * ratio)
            images1.append(self.data[i][0][:splitIndex])
            images2.append(self.data[i][0][splitIndex:])
            labels1.append(self.data[i][1][:splitIndex])
            labels2.append(self.data[i][1][splitIndex:])
        return DataGenerator(np.concatenate(images1, axis=0), np.concatenate(labels1, axis=0), self.batchCount), DataGenerator(np.concatenate(images2, axis=0), np.concatenate(labels2, axis=0), self.batchCount)
