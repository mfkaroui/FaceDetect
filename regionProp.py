import cv2 as cv
import numpy as np
import os
import psutil as psu
import functools
import tensorflow as tf

from cocoDataGenerator import DataGenerator
from cocoDataGenerator import shape as inputShape
import urllib.request
import shutil
import zipfile

import cv2 as cv

dataDir = "COCOData"
workingDir = os.getcwd()
dataDir = os.path.join(workingDir, dataDir)

dataType = 'val2017'
annDir = os.path.join(dataDir, "annotations")
annZipFile = os.path.join(dataDir, "annotations_train{}.zip".format(dataType))
annFile = os.path.join(annDir, "instances_{}.json".format(dataType))
annURL = "http://images.cocodataset.org/annotations/annotations_train{}.zip".format(dataType)

if __name__ == "__main__":
    print("Region Proposal\nAuthor: Mohamed Fateh Karoui")
    print("Working Directory : " + workingDir)
    ncpu = psu.cpu_count()
    print("Logical CPU's Found : " + str(ncpu))
    print("Looking for labels data...")
    if not os.path.exists(annDir):
        os.makedirs(annDir)
    if not os.path.exists(annFile):
        if not os.path.exists(annZipFile):
            print ("Downloading zipped annotations to " + annZipFile + " ...")
            with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print ("... done downloading.")
        print ("Unzipping " + annZipFile)
        with zipfile.ZipFile(annZipFile,"r") as zip_ref:
            zip_ref.extractall(dataDir)
        print ("... done unzipping")
    print ("Will use annotations in " + annFile)
    data = DataGenerator(annFile, 0.3, 500, 20)

    ConvTemplate = functools.partial(tf.keras.layers.Conv2D, padding="same", data_format="channels_last", bias_initializer=tf.keras.initializers.Zeros(), kernel_initializer=tf.keras.initializers.lecun_normal(), activation=tf.keras.activations.selu)
    ConvTemplate.__name__ = 'ConvTemplate'
    TConvTemplate = functools.partial(tf.keras.layers.Conv2DTranspose, padding="same", data_format="channels_last", bias_initializer=tf.keras.initializers.Zeros(), kernel_initializer=tf.keras.initializers.lecun_normal(), activation=tf.keras.activations.selu)
    TConvTemplate.__name__ = 'TConvTemplate'
    def skipStep(layer_1, layer_2):
        return tf.keras.layers.Concatenate()([layer_1, layer_2])

    layers = [tf.keras.layers.Input(shape=(inputShape[1], inputShape[0], 3))]
    dropoutRate = 0.0
    filterCount = 50
    kSize = 5

    kernelSizes = []
    filters = []
    dropoutRates = []
    for i in range(1, 5):
        kernelSizes.append((kSize, kSize))
        filters.append(filterCount)
        dropoutRates.append(dropoutRate)
        l = ConvTemplate(filters=filterCount, kernel_size=(kSize, kSize), strides=(2,2))(layers[i - 1])
        l = tf.keras.layers.SpatialDropout2D(dropoutRate, "channels_last")(l)
        dropoutRate = dropoutRate / 2
        filterCount = int(filterCount / 2)
        filterCount = 2 if filterCount < 2 else filterCount
        layers.append(l)
        kSize = int(kSize * 2)
        pShape = tf.keras.backend.int_shape(l)
        mSize = int(min(pShape[1], pShape[2]))
        kSize = kSize if kSize < mSize else mSize
        kSize = kSize - 1 if kSize % 2 == 0 else kSize
    #inverse operation with skipSteps
    for i in range(len(filters), 0, -1):
        l = TConvTemplate(filters=filters[i - 1], kernel_size=kernelSizes[i - 1], strides=(2,2))(layers[-1])
        if i > 3:
            l = skipStep(l, layers[i - 1])
        l = tf.keras.layers.SpatialDropout2D(min(dropoutRates[i - 1] * 2, 0.5), "channels_last")(l)
        layers.append(l)
    classification = ConvTemplate(filters=20, kernel_size=(3,3))(layers[-1])
    classification = ConvTemplate(filters=20, kernel_size=(3,3))(classification)
    classification = ConvTemplate(filters=2, kernel_size=(3,3), activation=tf.keras.activations.softmax)(classification)
    model = tf.keras.models.Model(inputs=layers[0], outputs=classification)


    def pr(b, l):
        preview = []
        images, labels = data.validationGenerator[0]
        n = 3
        predictions = model.predict_on_batch(images[:n])
        dummyChannel = np.zeros((inputShape[1], inputShape[0], 1))
        for j in range(n):
            row = [images[j], np.concatenate([labels[j], dummyChannel], axis=-1), np.concatenate([predictions[j], dummyChannel], axis=-1)]
            preview.append(np.concatenate(row, axis=-2))
        preview = np.concatenate(preview, axis=-3)
        cv.imshow("Preview", (preview * 255).astype(np.uint8))
        cv.waitKey(1)
    previewResults = tf.keras.callbacks.LambdaCallback(on_batch_begin=pr)

    for i in range(1000):
        model.compile(tf.keras.optimizers.Adam(1e-4),loss=data.pixelwise_crossentropy(), metrics=["accuracy"])
        data.fit(model, 100, [previewResults])
        data.sampleData()
