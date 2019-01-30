import tensorflow as tf
import numpy as np
import os
import psutil as psu
from multiprocessing import pool as mpp
from dataGenerator import DataGenerator
from lowLevelFeatures import EdgeFeatures2D

imagesPath = "Data/UTKFace"
labelsPath = "Data"

workingDir = os.getcwd()
imagesPath = os.path.join(workingDir, imagesPath)
labelsPath = os.path.join(workingDir, labelsPath)

genderClassLabels = ["Male", "Female"]
raceClassLabels = ["White", "Black", "Asian", "Indian", "Others"]
ageClassLabels = [(0, 5), (6, 11), (12, 17),
                  (18, 23), (24, 29), (30, 35),
                  (36, 41), (42, 47), (48, 53),
                  (54, 59), (60, 65), (66, 71),
                  (72, 77), (78, 83), (84, 89),
                  (90, 95), (96, 101), (102, 107),
                  (108, 113), (114, 119)]

def oneHotEncode(i, n):
    returnArray = np.zeros(shape=(n,), dtype=float)
    returnArray[i] = 1
    return returnArray

if __name__ == "__main__":
    print("Face Detect\nAuthor: Mohamed Fateh Karoui")
    print("Working Directory : " + workingDir)
    ncpu = psu.cpu_count()
    print("Logical CPU's Found : " + str(ncpu))
    print("Looking for labels data...")
    
    labels = {}
    for f in os.listdir(labelsPath):
        if os.path.isdir(f) == False and f.endswith(".txt"):
            with open(os.path.join(labelsPath, f), "r") as labelFile:
                labelFile.seek(0)
                content = labelFile.read()
                lines = content.split("\n")
                for line in lines:
                    parts = line.split(" ")
                    filenameParts = parts[0].split("_")
                    fullPath = os.path.join(imagesPath, parts[0] + ".chip.jpg")
                    if len(filenameParts) == 4 and os.path.exists(fullPath):
                        try:
                            age = int(filenameParts[0])
                            gender = int(filenameParts[1])
                            race = int(filenameParts[2])
                            labels[fullPath] = (age, gender, race)
                        except:
                            pass
    print("Found " + str(len(labels)) + " labels.")
    
    outputAges = []
    for label in list(labels.values()):
        for i in range(len(ageClassLabels)):
            if label[0] >= ageClassLabels[i][0] and label[0] <= ageClassLabels[i][1]:
                outputAges.append(oneHotEncode(i, len(ageClassLabels)))
                break
    outputAges = np.array(outputAges, dtype=float)

    print("Binned Ages into classes. Shape: " + str(outputAges.shape))

    data = DataGenerator(np.array(list(labels.keys())), outputAges, 5)
    
    inputLayer = tf.keras.layers.Input(shape=(200,200,3))
    hiddenLayer_1 = EdgeFeatures2D(20, (5,5), 0.8, 100, tf.keras.activations.relu)(inputLayer)
    hiddenLayer_2 = EdgeFeatures2D(20, (5,5), 0.8, 100, tf.keras.activations.relu)(hiddenLayer_1)
    hiddenLayer_3 = EdgeFeatures2D(20, (5,5), 0.8, 100, tf.keras.activations.relu)(hiddenLayer_2)
    hiddenLayer_4 = tf.keras.layers.Flatten()(hiddenLayer_3)
    hiddenLayer_5 = tf.keras.layers.Dense(units=outputAges.shape[1], activation=tf.keras.activations.softmax)(hiddenLayer_4)

    model = tf.keras.models.Model(inputs=inputLayer, outputs=hiddenLayer_5)
    model.compile(tf.keras.optimizers.Adam(1e-2),loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    model.fit_generator(generator=data, epochs=10000)
    print("Done")