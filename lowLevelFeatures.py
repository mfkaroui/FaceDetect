import tensorflow as tf
import numpy as np

#angle = angle of line in degrees
#sampleCount = number of points to sample for each slice of the line
#variance = correspondes to the thickness of the line usually (2 * variance) + 1
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
    result = np.zeros((height, width, depth), dtype=float)
    #the mean position initializes at the center
    mean = np.copy(center)
    while mean[0] >= 0 and mean[0] < height and mean[1] >= 0 and mean[1] < width:
        #draw random points that have a distance variance given by the argument
        distance = np.random.normal(scale=variance, size=(sampleCount, depth))
        for d in range(depth):
            for s in range(sampleCount):
                #calculate the position of the current point
                current = (mean + (deltaNormal * distance[s, d])).astype(int)
                #clamp the position if it exceeds the boundaries of the matrix
                current[0] = 0 if current[0] < 0 else height - 1 if current[0] >= height else int(current[0])
                current[1] = 0 if current[1] < 0 else width - 1 if current[1] >= width else int(current[1])
                result[current[0], current[1], d] += 1 / float(sampleCount * depth)
        mean += delta
    #now draw the other side of the line
    mean = np.copy(center) - delta
    while mean[0] >= 0 and mean[0] < height and mean[1] >= 0 and mean[1] < width:
        #draw random points that have a distance variance given by the argument
        distance = np.random.normal(scale=variance, size=(sampleCount, depth))
        for d in range(depth):
            for s in range(sampleCount):
                #calculate the position of the current point
                current = (mean + (deltaNormal * distance[s, d])).astype(int)
                #clamp the position if it exceeds the boundaries of the matrix
                current[0] = 0 if current[0] < 0 else height - 1 if current[0] >= height else int(current[0])
                current[1] = 0 if current[1] < 0 else width - 1 if current[1] >= width else int(current[1])
                result[current[0], current[1], d] += 1 / float(sampleCount * depth)
        mean -= delta
    return result# return the final tensor

def Edge2DInitializer(sampleCount, variance):
    def initializer(shape, dtype=None):
        totalEdges = shape[-1]
        edgeShape = shape[:-1]
        angleDelta = 180 / float(totalEdges)
        tensor = []
        for i in range(totalEdges):
            tensor.append(StochasticLine(i * angleDelta, sampleCount, variance, edgeShape[1], edgeShape[0], edgeShape[2]))
        tensor = tf.keras.backend.constant(np.stack(tensor, axis=-1), dtype)
        return tensor
    return initializer
#A Conv2D layer where the filters are preinitialized as edge features
class EdgeFeatures2D(tf.keras.layers.Layer):
    def __init__(self, filters, filterSize, filterVariance, filterSampleCount, activation, **kwargs):
        self.filters = filters
        self.filterSize = filterSize
        self.filterVariance = filterVariance
        self.filterSampleCount = filterSampleCount
        self.activation = activation
        super(EdgeFeatures2D, self).__init__(**kwargs)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        kernel_shape = self.filterSize + (input_channels, self.filters)
        bias_shape = (self.filters,)
        self.kernel = self.add_weight(name="kernel", shape=kernel_shape, initializer=Edge2DInitializer(self.filterSampleCount, self.filterVariance))
        self.bias = self.add_weight(name="bias", shape=bias_shape, initializer=tf.keras.initializers.Zeros)
        super(EdgeFeatures2D, self).build(input_shape)

    def call(self, x):
        return self.activation(tf.keras.backend.bias_add(tf.keras.backend.conv2d(x, self.kernel, data_format='channels_last'), self.bias, data_format='channels_last'))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)