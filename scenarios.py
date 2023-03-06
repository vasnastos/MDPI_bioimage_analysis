from models import ImageNet,OptunaModel,Controller
from itertools import product


def scenario1():
    imageNN=ImageNet()
    for pretrained_model_name,upper_bounds in Controller.freezing_layers.items():
        for ubound in upper_bounds:
            imageNN.extra=imageNN.extract_features(base_model_name=pretrained_model_name,lb=0,ub=ubound,save=True)

def scenario2():
    imageNN=ImageNet()
    optunahandler=OptunaModel(new_study_case="Study_scenario_2")
    for param,scope in {'accuracy_score':"maximize",'f1_score':"maximize",'cohens_kappa':"maximize"}.items():
        optunahandler.add_objective_dimension(param,scope)

    optunahandler.set_callback(imageNN.optuna_callback)
    optunahandler.export()
    optunahandler.export_best()

    


if __name__=='__main__':
    # scenario1() # 1. Extracted image features
    scenario2() # 2. Optimize the hyperparameters in the models
