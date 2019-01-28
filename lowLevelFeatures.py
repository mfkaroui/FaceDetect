import tensorflow as tf

class EdgeFeatures(tf.keras.layers.Layer):
    def __init__(self, count, size, variation, thickness, **kwargs):
        self.count = count
        self.size = size
        self.variation = variation
        self.thickness = thickness
        super(LowLevelFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = (int(input_shape[1] * self.size), int(input_shape[2] * self.size))
        super(LowLevelFeatures, self).build(input_shape)

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass