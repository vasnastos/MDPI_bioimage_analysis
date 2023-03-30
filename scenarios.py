import os,optuna,numpy as np,pickle

from models import ImageNet,Controller
from sklearn.model_selection import StratifiedKFold
from elayers import OptunaParamLayer
import pandas as pd

def scenario1():
    """
        Scenario for feature extraction model using:
             - Pretrained models: vgg-16,vgg19,resnet50,resnet101
    """
    imageNN=ImageNet()
    for pretrained_model_name,upper_bounds in Controller.freezing_layers.items():
        for ubound in upper_bounds:
            imageNN.extra=imageNN.extract_features(base_model_name=pretrained_model_name,lb=0,ub=ubound,save=True)


def scenario2():
    """
        Tune the selective fine tuning model using:
            - Pretrained models: vgg-16,vgg19,resnet50,resnet101
            - Freezing layers:  'vgg16': [6,10,14],
                                'vgg19': [6,11,16],
                                'resnet50': [39,85,153],
                                'resnet101': [39,85,340]
            - Optimizers: 'adam','sgd'
    """
    imageNN=ImageNet()
    Xentry,Yentry=imageNN.dataset.load_image_dataset(split=False)
    Xentry=np.array(Xentry)
    Yentry=np.array(Yentry)
    cv_model=StratifiedKFold(n_splits=5)
    OptunaParamLayer.clear_table()
    
    for train_indeces,test_indeces in cv_model.split(Xentry,Yentry):
        xtrain=np.array([Xentry[i] for i in train_indeces])
        xtest=np.array([Xentry[i] for i in test_indeces])
        ytrain=np.array([Yentry[i] for i in train_indeces])
        ytest=np.array([Yentry[i] for i in test_indeces])

        OptunaParamLayer.set_layer_data(xtrain,xtest,ytrain,ytest)
        study=optuna.create_study(directions=['maximize','maximize'])
        study.optimize(imageNN.optuna_callback2)
    
    imageNN.clear_session()
    with open(os.path.join('','results','pickle_ImageNET_Evaluation.pcl'),'wb') as bwriter:
        pickle.dump(OptunaParamLayer.results,bwriter)

def scenario3():
    """
        Try the conventional model using pretrained parameters
            - resnet101,0,340,Standardization,1633
    """
    imageNN=ImageNet()
    imageNN.preprocessing(feature_selection='gs')
    features=imageNN.bioimage_dataframe.columns.to_list()
    target=features[-1]
    features.remove(target)

    X,Y=imageNN.bioimage_dataframe[features],imageNN.bioimage_dataframe[target]
    cv_model=StratifiedKFold(n_splits=10)
    results=dict()
    for fold_no,(train_idx,test_idx) in enumerate(cv_model.split(X,Y)):
        xtrain=pd.DataFrame(data=[X.iloc[idx] for idx in train_idx],columns=X.columns.to_list())
        xtest=pd.DataFrame(data=[X.iloc[idx] for idx in test_idx],columns=X.columns.to_list())
        ytrain=pd.Series(data=[Y.iloc[idx] for idx in train_idx],name=Y.name)
        ytest=pd.Series(data=[Y.iloc[idx] for idx in test_idx],name=Y.name)
        results[fold_no]=imageNN.conventional_model(xtrain,xtest,ytrain,ytest,clf='all')
    
    print(results)




if __name__=='__main__':
    # scenario1() # 1. Extracted image features
    # scenario2() # 2. Evaluate the performance of the transfer learning model 
    scenario3()