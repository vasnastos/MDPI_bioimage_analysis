import tensorflow as tf,pandas as pd

class AddLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AddLayer,self).__init__()
    
    def call(self,inputs):
        return tf.keras.backend.sum(inputs,axis=1)


class OptunaParamLayer:
    xtrain=None
    xtest=None
    ytrain=None
    ytest=None
    results=dict()

    @staticmethod
    def set_layer_data(_xtrain,_xtest,_ytrain,_ytest):
        OptunaParamLayer.xtrain=_xtrain
        OptunaParamLayer.xtest=_xtest
        OptunaParamLayer.ytrain=_ytrain
        OptunaParamLayer.ytest=_ytest

    @staticmethod
    def clear_table():
        OptunaParamLayer.results.clear()
    
    @staticmethod
    def add(config,evaluation):
        if config not in OptunaParamLayer.results:
            OptunaParamLayer.results[config]=list()
        OptunaParamLayer[config].append(evaluation)

    @staticmethod
    def save_results(filepath,columns=None):
        pd.DataFrame(data=[list(config)+list(evaluation) for config,evaluation in OptunaParamLayer.results.items()],columns=columns).to_csv(filepath)
