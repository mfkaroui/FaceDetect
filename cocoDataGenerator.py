import numpy as np
import tensorflow as tf
import random

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import os
import psutil as psu
import urllib.request
import shutil
import zipfile
import itertools
import threading

shape = (512,256)

def getImg(url, result, index):
    global shape
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            img = Image.open(resp.fp)
            i = img.resize(shape, Image.ANTIALIAS)
            result[index] = np.asarray(i, dtype=float) / 255
            if len(result[index].shape) != 3 or result[index].shape[-1] != 3:
                result[index] = None
            i.close()
            img.close()
    except:
        result[index] = None
    return 0

def getLbl(annotation, width, height, personID, result, index):
    if result[index] is None:
        img = Image.new("RGB", (width, height), "black")
    if annotation["category_id"] == personID:
        draw = ImageDraw.Draw(img)
        if annotation["iscrowd"] == 1:
            draw.rectangle([(annotation["bbox"][0], annotation["bbox"][1]), (annotation["bbox"][0] + annotation["bbox"][2], annotation["bbox"][1] + annotation["bbox"][3])], fill=(255,255,255))
        else:
            for segment in annotation["segmentation"]:
                xy = []
                for i in range(1, len(segment), 2):
                    xy.append((segment[i - 1], segment[i]))
                draw.polygon(xy, fill=(255,255,255))
    i = img.resize(shape, Image.ANTIALIAS)
    result[index] = np.ceil(np.asarray(i, dtype=float)[:,:,0] / 255)
    i.close()
    img.close()
    return 0
'''
The data generator will be responsible for subdividing the input data into training sets and test sets
It will subsample classes per batch to solve class imbalance issues.
'''
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, annotationFile, validationSplit=0.3, samplesPerClass=50, batchCount=10, isValidation = False, validationPerson = None, validationBackground = None):

        #store the samplesPerClass
        self.samplesPerClass = samplesPerClass
        self.batchCount = batchCount
        #initialize coco api
        self.coco = COCO(annotationFile)
        self.isValidation = isValidation
        person = self.coco.getCatIds(catNms=['person'])
        self.personID = person[0]
        if isValidation == False:
            cats = self.coco.loadCats(self.coco.getCatIds())
            nms=[cat['name'] for cat in cats]
            nms.remove("person")

            self.person = self.coco.getImgIds(catIds=person)
            background = self.coco.getCatIds(catNms=nms)
            backgrounds = itertools.combinations(background, 3)
            self.background = []
            for b in backgrounds:
                self.background.extend(self.coco.getImgIds(catIds=b))
            for p in self.person:
                try:
                    self.background.remove(p)
                except:
                    pass
            #initial randomize
            self.person = np.array(self.person, dtype=int)
            self.background = np.array(self.background, dtype=int)

            i = random.sample(range(len(self.person)), len(self.person))
            self.person = self.person[i]
            personValidationIndex = int(self.person.shape[0] * (1 - validationSplit))
            personVal = self.person[personValidationIndex:]
            self.person = self.person[:personValidationIndex]
            i = random.sample(range(len(self.background)), len(self.background))
            self.background = self.background[i]
            backgroundValidationIndex = int(self.background.shape[0] * (1 - validationSplit))
            backgroundVal = self.background[backgroundValidationIndex:]
            self.background = self.background[:backgroundValidationIndex]
            self.validationGenerator = DataGenerator(annotationFile, validationSplit, samplesPerClass, batchCount, True, personVal, backgroundVal)
            self.sampleData()
        else:
            self.person = validationPerson
            self.background = validationBackground

    def sampleData(self):
        global shape
        self.images = []
        self.labels = []
        u = []
        personsIndex = random.sample(range(self.person.shape[0]), self.samplesPerClass)
        u.extend(self.person[personsIndex])
        backgroundIndex = random.sample(range(self.background.shape[0]), self.samplesPerClass)
        u.extend(self.background[backgroundIndex])

        u = self.coco.loadImgs(u)
        r = [None] * len(u)
        threads = [threading.Thread(target=getImg, args=(url["coco_url"],r,u.index(url))) for url in u]
        for thread in threads:
            thread.start()
        while True:
            recheck = False
            for thread in threads:
                if thread.isAlive():
                    recheck = True
            if recheck == False:
                break
        i = 0
        totalPersonPixels = 0
        for res in r:
            if res is not None:
                self.images.append(res)
                a = self.coco.getAnnIds(imgIds=[u[i]["id"]], iscrowd=None)
                a = self.coco.loadAnns(a)
                r1 = [None] * len(a)
                threads = [threading.Thread(target=getLbl, args=(annotation, u[i]["width"], u[i]["height"], self.personID, r1, a.index(annotation))) for annotation in a]
                for thread in threads:
                    thread.start()
                while True:
                    recheck = False
                    for thread in threads:
                        if thread.isAlive():
                            recheck = True
                    if recheck == False:
                        break
                r1 = np.stack(r1, axis=0)
                label = np.amax(r1, axis=0)
                totalPersonPixels += np.sum(label)
                label = np.stack([label, 1 - label], axis=2)
                self.labels.append(label)
            i = i + 1
        personWeight = 1 - (totalPersonPixels / (shape[0] * shape[1] * len(self.images)))
        self.classWeights = tf.keras.backend.constant(np.array([personWeight, 1 - personWeight]))
        self.images = np.stack(self.images, axis=0)
        self.labels = np.stack(self.labels, axis=0)

        #final shuffle
        indexes = random.sample(range(self.images.shape[0]), self.images.shape[0])
        self.images = self.images[indexes]
        self.labels = self.labels[indexes]

        self.len = int(self.images.shape[0] / self.batchCount)

        if self.isValidation == False:
            self.validationGenerator.sampleData()

    def fit(self, model, epochs, callbacks=[]):
        model.fit_generator(generator=self, validation_data=self.validationGenerator, epochs=epochs, callbacks=callbacks)

    def __len__(self):
        return self.batchCount

    def __getitem__(self, index):
        s = slice(index * self.len, (index + 1) * self.len)
        return self.images[s], self.labels[s]

    def on_epoch_end(self):
        #when we finish an epoch we will shuffle
        indexes = random.sample(range(self.images.shape[0]), self.images.shape[0])
        self.images = self.images[indexes]
        self.labels = self.labels[indexes]

    def pixelwise_crossentropy(self):
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
            return - tf.reduce_sum(tf.multiply(y_true * tf.keras.backend.log(y_pred), self.classWeights))
        return loss
