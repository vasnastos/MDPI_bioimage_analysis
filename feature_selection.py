import tensorflow as tf
import copy,numpy as np,random,optuna
import os,pandas as pd
import oapackage
from tabulate import tabulate

from rich.console import Console
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,cohen_kappa_score,f1_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as RFE

import matplotlib.pyplot as plt

class LassoSelector:
    def __init__(self,dataset_id,x_train,x_test,y_train,y_test,pretrained_model):
        self.id=dataset_id
        self.xtrain=x_train
        self.xtest=x_test
        self.ytrain=y_train
        self.ytest=y_test
        self.model_name=pretrained_model
        self.configurations=dict()
        
    def callback(self,trial):
        normalize=trial.suggest_categorical('scaling_method',['MinMax','Standardization','KnnImpute'])
        alpha_param=trial.suggest_categorical('lasso_alpha',[str(x) for x in np.arange(0.5,20,0.5)])
        
        scaler=MinMaxScaler() if normalize=='MinMax' else StandardScaler() if normalize=='Standardization' else KNNImputer(n_neighbors=10)
        lasso_pipeline = Pipeline(
        steps=[
            ('scaler', scaler),
            ('lasso', Lasso(alpha=float(alpha_param),tol=1e-2,max_iter=10000))
        ],verbose=True)

        lasso_pipeline.fit(self.xtrain,self.ytrain)
        ypred=lasso_pipeline.predict(self.xtest)
        mse,r2=mean_squared_error(self.ytest,ypred),r2_score(self.ytest,ypred)
        feature_coefficients=lasso_pipeline.named_steps['lasso'].coef_
        selected_columns=[column for i,column in enumerate(self.xtrain.columns.to_list()) if feature_coefficients[i]!=0]
        self.configurations[(normalize,alpha_param)]=(len(selected_columns),mse,r2)

        return mse

    def optimize(self):
        study=optuna.create_study(direction='minimize')
        study.optimize(self.callback,n_trials=50)

        lb,ub=self.id.split('_')[3],self.id.split('_')[4]

        file_exists=os.path.exists(os.path.join('','results','optuna','LassoSelector',f'selector_optimization_{self.model_name}.csv'))
        with open(os.path.join('','results','optuna','LassoSelector',f'selector_optimization_{self.model_name}.csv'),'a') as writer:
            if not file_exists:
                writer.write('Model,Lower Bound,Upper Bound,Scaling method,Alpha,Mean Square Error,R2-Square\n')
            for (scaling,alpha),(sfn,mse,r2s) in self.configurations.items():
                writer.write(f'{self.model_name},{lb},{ub},{scaling},{alpha},{sfn},{mse},{r2s}\n')

        return study.best_params

    def select(self,normalize,alpha):
        scaler=MinMaxScaler() if normalize=='MinMax' else StandardScaler() if normalize=='Standardization' else KNNImputer(n_neighbors=10)
        lasso_pipeline = Pipeline(
        steps=[
            ('scaler', scaler),
            ('lasso', Lasso(alpha=alpha,tol=1e-2,max_iter=10000))
        ],verbose=True)

        lasso_pipeline.fit(self.xtrain,self.ytrain)
        feature_coefficients=lasso_pipeline.named_steps['lasso'].coef_
        selected_columns=[column for column in self.xtrain.columns.to_list() if feature_coefficients[column]!=0]
        return (self.xtrain[selected_columns],self.xtest[selected_columns])

class GeneticSelector:
    def __init__(self,_xtrain,_xtest,_ytrain,_ytest,pretrained_model):
        self.xtrain=_xtrain
        self.xtest=_xtest
        self.ytrain=_ytrain
        self.ytest=_ytest
        self.configurations=dict()
        self.model_name=pretrained_model
    
    def solve(self,trial):
        scaling_method=trial.suggest_categorical('scaling',["MinMax","z-Score","knn-Impute"])
        number_of_selected_features=trial.suggest_categorical('number_of_selected_features',list(range(self.xtrain.shape[1]//2,self.xtrain.shape[1])))

        # 1. Scaling procedure
        scaler=MinMaxScaler() if scaling_method=="MinMax" else StandardScaler() if scaling_method=='z-Score' else KNNImputer(n_neighbors=10)
        self.xtrain=scaler.fit_transform(self.xtrain,self.ytrain)
        self.xtest=scaler.fit_transform(self.xtest,self.ytest)

        # 2. Genetic Selection
        estimator= AdaBoostClassifier(learning_rate=1e-3,n_estimators=100)
        model=GeneticSelectionCV(
            estimator=estimator,
            cv=10,
            max_features=number_of_selected_features,
            scoring='accuracy',
            n_population=50,
            crossover_independent_proba=0.5,
            mutation_proba=0.2,
            n_generations=50,
            mutation_independent_proba=0.04,
            tournament_size=5,
            n_gen_no_change=10,
            n_jobs=-1,
            verbose=True        
        )
        model=model.fit(self.xtrain,self.ytrain)
        self.xtrain=self.xtrain[model.get_feature_names_out()]
        self.xtest=self.xtest[model.get_feature_names_out()]

        estimateModel=RandomForestClassifier(n_estimators=100,class_weight='balanced')
        estimateModel=estimateModel.fit(self.xtrain,self.ytrain)
        predictions=estimateModel.predict(self.xtest)
        acc_score,f1=accuracy_score(self.ytest,predictions),f1_score(self.ytest,predictions)
        self.configurations[(scaling_method,number_of_selected_features)]=(acc_score,f1)
        return acc_score,f1

    def optimize(self):
        study=optuna.create_study(directions=['maximize','maximize'])
        study.optimize(self.solve,n_trials=50)

        file_exists=os.path.exists(os.path.join('','results','optuna','GeneticSelector',f'selector_optimization_{self.model_name}.csv'))
        with open(os.path.join('','results','optuna','GeneticSelector',f'selector_optimization_{self.model_name}.csv'),'a') as writer:
            if not file_exists:
                writer.write('Model','Scaling method','Number of Features','Accuracy','F1-Score\n')
            for (scaling,number_of_features),(acc_score,f1score) in self.configurations.items():
                writer.write(f'{self.model_name},{scaling},{number_of_features},{acc_score},{f1score}\n')
        
        return study.best_params

    def select(self,scaling_method,number_of_features):
        scaler=MinMaxScaler() if scaling_method=="MinMax" else StandardScaler() if scaling_method=='z-Score' else KNNImputer(n_neighbors=10)
        self.xtrain=scaler.fit_transform(self.xtrain,self.ytrain)
        self.xtest=scaler.fit_transform(self.xtrain,self.ytrain)

        # 2. Genetic Selection
        estimator= AdaBoostClassifier(learning_rate=1e-3,n_estimators=100)
        model=GeneticSelectionCV(
            estimator=estimator,
            cv=10,
            max_features=number_of_features,
            scoring='accuracy',
            n_population=100,
            crossover_independent_proba=0.5,
            mutation_proba=0.2,
            n_generations=100,
            mutation_independent_proba=0.04,
            tournament_size=5,
            n_gen_no_change=10,
            n_jobs=-1,
            verbose=True        
        )
        model=model.fit(self.xtrain,self.ytrain)
        self.xtrain=self.xtrain[model.get_feature_names_out()]
        self.xtest=self.xtest[model.get_feature_names_out()]
    
    def get_params(self):
        return self.xtrain,self.xtest

class PCASelector:
    def __init__(self,dataset_id,_xtrain,_xtest,_ytrain,_ytest) -> None:
        self.xtrain=_xtrain
        self.xtest=_xtest
        self.ytrain=_ytrain
        self.ytest=_ytest
        self.configurations=dict()
        self.pretrained_model_name=dataset_id.split('_')[2]
        self.lb,self.ub=int(dataset_id.split('_')[3]),int(dataset_id.split('_')[4].removesuffix('.csv'))

    def callback(self,trial):
        scaling_method=trial.suggest_categorical('scaling',['MinMax','Standardization'])
        nof=trial.suggest_categorical('feature_selection',[str(x) for x in range(len(self.xtrain.columns.to_list())//2,min(len(self.xtrain.columns.to_list()),self.xtrain.shape[0]))])
        scaler=MinMaxScaler() if scaling_method=='MinMax' else StandardScaler() 
        pipe=Pipeline(
            steps=[
                ('scaler',scaler),
                ('PCA',PCA(n_components=int(nof),svd_solver='full')),
                ('Ada',AdaBoostClassifier(learning_rate=1e-2))
            ],
            verbose=True
        )

        pipe=pipe.fit(self.xtrain,self.ytrain)
        ypred=pipe.predict(self.xtest)

        acc_score,f1,cohens=accuracy_score(self.ytest,ypred),f1_score(self.ytest,ypred,average='macro'),cohen_kappa_score(self.ytest,ypred)
        self.configurations[(scaling_method,nof)]=(acc_score,f1,cohens)
        return acc_score,f1

    def optimize(self):
        study=optuna.create_study(directions=["maximize","maximize"])
        study.optimize(self.callback,n_trials=50)

        file_exists=os.path.exists(os.path.join('','results','optuna','PCA',f'selector_optimization_{self.pretrained_model_name}.csv'))
        with open(os.path.join('','results','optuna','PCA',f'selector_optimization_{self.pretrained_model_name}.csv'),'a') as writer:
            if not file_exists:
                writer.write('Pretrained model,Lower Bound,Upper bound,Scaling,Number of features,Accuracy Score,F1 Score,Cohens Kappa Score\n')
            for (scaling,nof),(acc_score,f1,cohens) in self.configurations.items():
                writer.write(f'{self.pretrained_model_name},{self.lb},{self.ub},{scaling},{nof},{acc_score},{f1},{cohens}\n')

def pareto(fselector):
    subfolder=None
    metrics=[]
    if fselector=='gs':
        subfolder='GeneticSelector'
        metrics=[
            'Accuracy Score',
            'F1 Score'
        ]
    elif fselector=='lasso':
        subfolder='LassoSelector'
        metrics=[
            'Mean Square Error',
            'R2-Square'
        ]
    elif fselector=='pca':
        subfolder='PCA'
        metrics=[
            'Accuracy Score',
            'F1 Score'
        ]
    
    # Pretrained model	Lower Bound	Upper bound	Scaling
    results=list()
    for resfile in os.listdir(os.path.join('','results','optuna',subfolder)):
        results.append(pd.read_csv(os.path.join('','results','optuna',subfolder,resfile),header=[0]))

    pareto_source=pd.concat(results)

    pareto_objects=oapackage.ParetoDoubleLong()
    for i in range(pareto_source.shape[0]):
        pareto_objects.addvalue(oapackage.doubleVector((pareto_source.iloc[i][metrics[0]],pareto_source.iloc[i][metrics[1]])),i)
    pareto_objects.show(verbose=True)
    selected_indeces=pareto_objects.allindices()
    pareto_objects_final=pareto_source.iloc[list(selected_indeces),:]
    rows=[]
    for i in range(pareto_objects_final.shape[0]):
        primary_df_index=pareto_objects_final.index[i]
        if fselector=='lasso':
            rows.append([pareto_source.iloc[primary_df_index]['Pretrained model'],pareto_source.iloc[primary_df_index]['Lower Bound'],pareto_source.iloc[primary_df_index]['Upper bound'],pareto_source.iloc[i]['Scaling'],pareto_source.iloc[i]['alpha'],pareto_source.iloc[i]['Features'],pareto_objects_final.iloc[i][metrics[0]],pareto_objects_final.iloc[i][metrics[1]]])
            column_names=['Pretrained model','Lower Bound','Upper bound','Scaling','Alpha','Features']
            column_names.extend(metrics)
        else:
            rows.append([pareto_source.iloc[primary_df_index]['Pretrained model'],pareto_source.iloc[primary_df_index]['Lower Bound'],pareto_source.iloc[primary_df_index]['Upper bound'],pareto_source.iloc[i]['Scaling'],pareto_source.iloc[i]['Features'],pareto_objects_final.iloc[i][metrics[0]],pareto_objects_final.iloc[i][metrics[1]]])
            column_names=['Pretrained model','Lower Bound','Upper bound','Scaling','Features']
            column_names.extend(metrics)
    
    pareto_optimal_data=pd.DataFrame(rows,columns=column_names)
    pareto_optimal_data.to_csv(os.path.join('','results',f'{subfolder}_pareto_optimal_objects.csv'))
    print(tabulate(rows,headers=column_names,tablefmt='fancy_grid'))

    with open(os.path.join('','results',f'{subfolder}_pareto_optimal_objects.tex'),'w') as writer:
        writer.write(tabulate(rows,headers=column_names,tablefmt='latex'))


def scenario1():
    for dataset in os.listdir(os.path.join('','results','extracted_features')):
        data=pd.read_csv(os.path.join(os.path.join('','results','extracted_features',dataset)),header=[0])
        cv_model=StratifiedKFold(n_splits=10)
        features=data.columns.to_list()
        target=features[-1]
        features.remove(target)
        X,Y=data[features],data[target]
        pretrained_model=dataset.split("_")[2]

        for train_set,test_set in cv_model.split(X,Y):
            xtrain=pd.DataFrame(data=[X.iloc[i][features].to_list() for i in train_set],columns=X.columns.to_list())
            xtest=pd.DataFrame(data=[X.iloc[i][features].to_list() for i in test_set],columns=X.columns.to_list())
            ytrain=pd.Series(data=[Y.iloc[i] for i in train_set],name=target)
            ytest=pd.Series(data=[Y.iloc[i] for i in test_set],name=target)

            selector=GeneticSelector(xtrain,xtest,ytrain,ytest,pretrained_model)
            print(selector.optimize())

def scenario2():
    for dataset in os.listdir(os.path.join('','results','extracted_features')):
        data=pd.read_csv(os.path.join(os.path.join('','results','extracted_features',dataset)),header=[0])
        cv_model=StratifiedKFold(n_splits=10)
        features=data.columns.to_list()
        target=features[-1]
        features.remove(target)
        X,Y=data[features],data[target]
        pretrained_model=dataset.split("_")[2]

        for train_set,test_set in cv_model.split(X,Y):
            xtrain=pd.DataFrame(data=[X.iloc[i][features].to_list() for i in train_set],columns=X.columns.to_list())
            xtest=pd.DataFrame(data=[X.iloc[i][features].to_list() for i in test_set],columns=X.columns.to_list())
            ytrain=pd.Series(data=[Y.iloc[i] for i in train_set],name=target)
            ytest=pd.Series(data=[Y.iloc[i] for i in test_set],name=target)

            selector=LassoSelector(dataset,xtrain,xtest,ytrain,ytest,pretrained_model)
            print(selector.optimize())

def scenario3():
     for dataset in os.listdir(os.path.join('','results','extracted_features')):
        data=pd.read_csv(os.path.join(os.path.join('','results','extracted_features',dataset)),header=[0])
        cv_model=StratifiedKFold(n_splits=10)
        features=data.columns.to_list()
        target=features[-1]
        features.remove(target)
        X,Y=data[features],data[target]

        for train_set,test_set in cv_model.split(X,Y):
            xtrain=pd.DataFrame(data=[X.iloc[i][features].to_list() for i in train_set],columns=X.columns.to_list())
            xtest=pd.DataFrame(data=[X.iloc[i][features].to_list() for i in test_set],columns=X.columns.to_list())
            ytrain=pd.Series(data=[Y.iloc[i] for i in train_set],name=target)
            ytest=pd.Series(data=[Y.iloc[i] for i in test_set],name=target)

            selector=PCASelector(dataset,xtrain,xtest,ytrain,ytest)
            selector.optimize()

def scenario4():
    pareto(fselector='gs')
    # pareto(fselector='lasso')
    # pareto(fselector='pca')

if __name__=='__main__':
    # scenario1() # Genetic Selector
    # scenario2() # Lasso Extraction Selector
    # scenario3() # PCA feature selection
    scenario4()  # Plot pareto front results
    




