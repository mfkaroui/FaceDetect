import tensorflow as tf
import numpy as np

#angle = angle of line in degrees
#sampleCount = number of points to sample for each slice of the line
#variance = correspondes to the thickness of the line
#width = width of the tensor (columns)
#height = height of the tensor (rows)
#depth = depth of the tensor (number of slices)
def StochasticLine(angle, sampleCount, variance, width, height, depth):
    #Calculate the center of the matrix, this will act as the origin of the line
    center = np.array([height / 2, width / 2])
    #Calculate the angle of the line
    angle = np.deg2rad(angle)
    #Calculate the normal line angle
    normalAngle = angle + (np.pi / 2)
    #calculate the delta spacing between each point on the line
    delta = np.array([np.cos(angle), np.sin(angle)])
    #calculate the normal delta, this will be used for the sampleCount of the line
    deltaNormal = np.array([np.cos(normalAngle), np.sin(normalAngle)])
    result = []
    #iterate through each slice in the depth of the tensor
    for d in range(depth):
        #preallocate the matrix for the slice
        m = np.zeros((height, width, 1), dtype=float)
        #the mean position initializes at the center
        mean = center
        while mean[0] >= 0 and mean[0] <= (height - 1) and mean[1] >= 0 and mean[1] <= (width - 1):
            #draw random points that have a distance variance
            distance = np.random.normal(scale=variance, size=(sampleCount,))
            for t in range(sampleCount):
                #calculate the position of the current point
                current = (mean + (deltaNormal * distance[t])).astype(int)
                #clamp the position if it exceeds the boundaries of the matrix
                current[0] = 0 if current[0] < 0 else height - 1 if current[0] >= height else int(current[0])
                current[1] = 0 if current[1] < 0 else width - 1 if current[1] >= width else int(current[1])
                m[current[0], current[1], 0] += 1 / float(sampleCount)
            mean += delta
        mean = center - delta
        while mean[0] >= 0 and mean[0] <= (height - 1) and mean[1] >= 0 and mean[1] <= (width - 1):
            #draw random points that have a distance variance from the current vector equal to the sampleCount of the line
            distance = np.random.normal(scale=variance, size=(sampleCount,))
            for t in range(sampleCount):
                #calculate the position of the current point
                current = (mean + (deltaNormal * distance[t])).astype(int)
                #clamp the position if it exceeds the boundaries of the matrix
                current[0] = 0 if current[0] < 0 else height - 1 if current[0] >= height else int(current[0])
                current[1] = 0 if current[1] < 0 else width - 1 if current[1] >= width else int(current[1])
                m[current[0], current[1], 0] += 1 / float(sampleCount)
            mean -= delta
        result.append(m)
    return np.concatenate(result, axis=-1)#stack the slices and return the final tensor

class EdgeFeatures(tf.keras.layers.Layer):
    def __init__(self, count, size, variation, sampleCount, **kwargs):
        self.count = count
        self.size = size
        self.variation = variation
        self.sampleCount = sampleCount
        super(EdgeFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = (int(input_shape[1] * self.size), int(input_shape[2] * self.size))
        super(EdgeFeatures, self).build(input_shape)

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass