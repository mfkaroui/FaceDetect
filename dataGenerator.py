import numpy as np
import tensorflow as tf
import random


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
        
    def sampleData(self):
        images = []
        labels = []
        #iterate through each class
        for i in range(self.nclasses):
            #select from each class randomly
            indexes = random.sample(range(self.data[i][0].shape[0]), self.maxSampleSize)
            images.extend(self.data[i][0][indexes])
            labels.extend(self.data[i][1][indexes])
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        #final shuffle
        indexes = random.sample(range(self.maxSampleSize * self.nclasses), self.maxSampleSize * self.nclasses)
        images = images[indexes]
        labels = labels[indexes]
        return images, labels
    
    def __len__(self):
        return self.batchCount
    
    def __getitem__(self, index):
        return self.sampleData()