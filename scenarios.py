import os,optuna,numpy as np,pickle

from models import ImageNet,Controller
from sklearn.model_selection import StratifiedKFold
from elayers import OptunaParamLayer



def scenario1():
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
    cv_model=StratifiedKFold(n_splits=5,test_size=0.3)
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
    # OptunaParamLayer.save_results(os.path.join('','results','optuna','ImageNet'),columns=['Pretrained model','Upper bound','Optimizer','Accuracy','F1 Score'])
    with open(os.path.join('','results','pickle_ImageNET_Evaluation.pcl'),'wb') as bwriter:
        pickle.dump(OptunaParamLayer.results,bwriter)

if __name__=='__main__':
    # scenario1() # 1. Extracted image features
    scenario2() # 2. Evaluate the performance of the transfer learning model 