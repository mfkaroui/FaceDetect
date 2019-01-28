import tensorflow as tf
import numpy as np
import os
import psutil as psu
from multiprocessing import pool as mpp
import PIL

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

def readImage(path):
    return np.asarray(PIL.Image.open(path))

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

    images = []
    for path in list(labels.keys()):
        images.append(readImage(path))

    images = np.array(images, dtype=float)

    print("Loaded Images. Shape: " + str(images.shape))

    inputImages = images / 255
    print("Normalized Input Image Data")

    outputAges = np.array([label[0] for label in list(labels.values())], dtype=int)
    