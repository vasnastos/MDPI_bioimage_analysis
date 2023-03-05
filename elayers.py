import tensorflow as tf

class AddLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AddLayer,self).__init__()
    
    def call(self,inputs):
        return tf.keras.backend.sum(inputs,axis=1)