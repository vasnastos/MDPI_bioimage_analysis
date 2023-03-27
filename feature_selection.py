import tensorflow as tf
import copy,numpy as np,random,optuna
from rich.console import Console
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score,r2_score,make_scorer,mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
import os,pandas as pd


class GeneticSelector:
    """
        Deprecated: Genetic Selector from sklearn used eventually
    """
    
    def __init__(self,x_train,x_test,y_train,y_test):
        self.xtrain=x_train
        self.xtest=x_test
        self.ytrain=y_train
        self.ytest=y_test

        self.xtrain=np.asarray(self.xtrain).astype(np.float64)
        self.xtest=np.asarray(self.xtest).astype(np.float64)
        self.ytrain=np.asarray(self.ytrain).astype(np.float64)
        self.ytest=np.asarray(self.ytest).astype(np.float64)

        self.model=None
    
    def get_n_individual(self,counter,population):
        """
            if counter is 0, return the individual with the highest prob
            if counter is 1, return the  second individual with the highest prob
            if counter is 2, return the third individual with highest prob
        """
        index=counter+1
        propabilities=[ind[1] for ind in population]
        sorted_propabilities=sorted(propabilities,key=float)
        max_prob=sorted_propabilities[-index]
        return [ind[0] for ind in population if ind[1]==max_prob]

    def generate_individuals(self,num_individuals:int,num_features:int,max_features=None):
        """
        Randomly selected individuals
        - num_individuals: int
            number of individuals to be generated
        - num_features: int
            define the length of each individual
        - max_features: int
            The maximum number of active features at each individual
        """
        individuals=list()
        for _ in range(num_individuals):
            individual=''
            for _ in range(num_features):
                if individual.count('1')==max_features:
                    individual+='0'
                    continue
                
                individual+=str(random.randint(0,1))
            individuals.append(individual)
        return individuals

    def create_model(self):
        self.model=tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(32,input_dim=self.xtrain.shape[1],activation='relu'))
        self.model.add(tf.keras.layers.Dense(64,activation='relu'))
        self.model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

        self.console.print(f'[bold red]{self.model.summary()}')
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        

    def get_fitness_func(self,individual:int):
        """
        Calculate the accuracy for the individual passed as parameter
        - individual: int
            number of individual calculate 
        """
        if self.model==None:
            raise AttributeError("Model does not set to None type")
        
        pred=self.model.predict(self.xtest)
        return accuracy_score(self.ytest,pred)

    def get_weights(self,population):
        total_accuracies=sum([individual[1] for individual in population])
        return [(individual[0],float((individual[1]/total_accuracies)*100)) for individual in population]

    def fill_population(self,individuals,goal_state:float):
        population=list()
        for individual in individuals:
            accuracy=self.get_fitness_func(individual)
            if accuracy>goal_state:
                return individual

            individual_complete=(individual,accuracy)
            population.append(individual_complete)
        return self.get_weights(population)

    def choose_parents(self,population,counter):
        """
            From the population, weighting the probabilities of an individual being chosen via the fitness
            function, takes randomly two individual to reproduce
            - population: list[tuple] 
                * The first element is the individual 
                * The second one is the probability associated to it.
            - counter: int
            To avoid generating repeated individuals, 'counter' parameter is used to pick parents in different ways, thus
            generating different individuals
        """
        if counter==0:
            parent1=self.get_n_individual(0,population=population)
            parent2=self.get_n_individual(1,population=population)
        elif counter==1:
            parent1=self.get_n_individual(0,population=population)
            parent2=self.get_n_individual(2,population=population)
        else:
            propabilities=(individual[1] for individual in population)
            individuals=[individual[0] for individual in population]
            parent1,parent2=random.choices(individuals,weights=propabilities,k=2)
        
        return [parent1,parent2]

    def mutate(self,child,prob=0.1):
        """
            Randomly mutates an individual according to the probability given by prob parameter
        """
        new_child=copy.deepcopy(child)
        for i,charv in enumerate(new_child):
            if random.random()<prob:
                new_value='1' if charv=='0' else '0'
                new_child=new_child[:i]+new_value+new_child[i+1:]
        return new_child

    def reproduce(self,individual_1, individual_2):
        """
        Takes 2 individuals, and combines their information based on a
        randomly chosen crosspoint.
        Each reproduction returns 2 new individuals
        """ 
        crosspoint=random.randint(1,len(individual_1)-1)
        child_1=individual_1[:crosspoint]-individual_2[crosspoint:]
        child_2=individual_2[:crosspoint]-individual_1[crosspoint:]

        return [self.mutate(child_1),self.mutate(child_2)]

    def generate_ahead(self,population):
        """
        Reproduces all the steps for choosing parents and making 
        childs, which means creating a new generation to iterate with
        """
        new_population=list()
        for _ in range(int(len(population))//2):
            parent1,parent2=self.choose_parents(population,counter=_)
            children=self.reproduce(parent1,parent2)
            new_population.append(children)
        return new_population

    def solve(self,ind_num,number_of_features,max_iter=5):
        console=Console(record=True)
        """
            Performs all the steps of the Genetic Algorithm
            1. Generate random population
            2. Fill population with the weights of each individual
            3. Check if the goal state is reached
            4. Reproduce the population, and create a new generation
            5. Repeat process until termination condition is met
        """

        individuals=self.generate_individuals(ind_num,number_of_features,10)
        population=self.fill_population(individuals,0.8)
        if isinstance(population,str):
            return population
        
        new_generation=self.generation_ahead(population)
        iter_cnt=0
        while iter_cnt<max_iter:
            console.print(f'[bold green] Iteration number{iter_cnt}\tIter_max:{max_iter}')
            population=self.fill_population(new_generation,0.8)

            if isinstance(population,str):
                break
            new_generation=self.generation_ahead(population)
            iter_cnt+=1
        return population

class LassoSelector:
    def __init__(self,dataset_id,x_train,x_test,y_train,y_test,pretrained_model):
        self.id
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
        self.configurations[(normalize,alpha_param)]=mean_squared_error(self.ytest,ypred),r2_score(self.ytest,ypred)

        return mean_squared_error(self.ytest,ypred)

    def optimize(self):
        study=optuna.create_study(direction='minimize')
        study.optimize(self.callback,n_trials=50)

        file_exists=os.path.exists(os.path.join('','results','optuna','LassoSelector',f'selector_optimization_{self.model_name}.csv'))
        with open(os.path.join('','results','optuna','LassoSelector',f'selector_optimization_{self.model_name}.csv'),'a') as writer:
            if not file_exists:
                writer.write('Model,Lower Bound,Upper Bound,Scaling method,Number of Features,Accuracy,F1-Score\n')
            for (scaling,number_of_features),(mse,r2s) in self.configurations.items():
                writer.write(f'{self.model_name},{scaling},{number_of_features},{mse},{r2s}\n')

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
        self.xtest=scaler.fit_transform(self.xtrain,self.ytrain)

        # 2. Genetic Selection
        estimator= AdaBoostClassifier(learning_rate=1e-3,n_estimators=100)
        model=GeneticSelectionCV(
            estimator=estimator,
            cv=10,
            max_features=number_of_selected_features,
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

            selector=LassoSelector(xtrain,xtest,ytrain,ytest,pretrained_model)
            print(selector.optimize())

if __name__=='__main__':
    scenario1() # Genetic Selector
    # scenario2() # Lasso Extractibon Selector
    




