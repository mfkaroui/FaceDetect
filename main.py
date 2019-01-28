#import tensorflow as tf
import numpy as np
import os
import psutil as psu
from multiprocessing import pool as mpp

imagesPath = "Data/UTKFace/"
labelsPath = "Data/"

workingDir = os.getcwd()
imagesPath = os.path.join(workingDir, imagesPath)
labelsPath = os.path.join(workingDir, labelsPath)


if __name__ == "__main__":
    print("Face Detect\nAuthor: Mohamed Fateh Karoui")
    print("Working Directory : " + workingDir)
    ncpu = psu.cpu_count()
    print("Logical CPU's Found : " + str(ncpu))
    print("Looking for labels data...")
    labelFiles = []
    for f in os.listdir(labelsPath):
        if os.path.isdir(f) == False and f.endswith(".txt"):
            labelFiles.append(os.path.join(labelsPath, f))
    print("Found " + str(len(labelFiles)) + " label files.")