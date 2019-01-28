import tensorflow as tf
import numpy as np
import os
import psutil as psu
from multiprocessing import pool as mpp
import PIL

imagesPath = "Data/UTKFace/"
labelsPath = "Data/"

workingDir = os.getcwd()
imagesPath = os.path.join(workingDir, imagesPath)
labelsPath = os.path.join(workingDir, labelsPath)

genderClassLabels = ["Male", "Female"]
raceClassLabels = ["White", "Black", "Asian", "Indian", "Others"]

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
                    if len(filenameParts) == 4:
                        try:
                            age = int(filenameParts[0])
                            gender = int(filenameParts[1])
                            race = int(filenameParts[2])
                            labels[parts[0]] = (age, gender, race)
                        except:
                            pass
    print("Found " + str(len(labels)) + " labels.")