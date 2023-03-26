from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import AdaBoostClassifier
import copy,optuna,random,numpy as np,pandas as pd,os,time,sys
from rich.console import Console
from sklearn.model_selection import  StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import statistics

class GSParam:
    population=None
    acc_score=-1
    results=[]

    @staticmethod
    def add_result(new_population_set,accuracy_score):
        GSParam.results.append((new_population_set,accuracy_score)) 
    
    @staticmethod
    def best_population(candicate_best_population,candicate_acc_score):
        if GSParam.population==None:
            GSParam.population=candicate_best_population
            GSParam.acc_score=candicate_acc_score
        else:
            if candicate_acc_score>GSParam.acc_score:
                GSParam.population=candicate_best_population
                GSParam.acc_score=candicate_acc_score

class GeneticSelector:
    def __init__(self,x_train,x_test,y_train,y_test,normalize='MinMax'):
        self.xtrain=x_train
        self.xtest=x_test
        self.ytrain=y_train
        self.ytest=y_test
        self.dimension=self.xtrain.shape[1]
        self.scaler=MinMaxScaler() if normalize=='MinMax' else StandardScaler() if normalize=='Standardization' else KNNImputer(n_neighbors=10)
        self.model=None

    def get_n_individual(self,counter,population):
        index=counter+1
        propabilities=[ind[1] for ind in population]
        sorted_probs=sorted(propabilities,key=float)
        max_prob=sorted_probs[-index]
        return [ind[0] for ind in population if ind[1]==max_prob][0]

    # tune max features number
    def generate_random_individuals(self,num_individuals,num_features,max_features=None):
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
        print(f'{self.xtrain.shape=}')
        if self.model:
            return
        self.model=tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(32,input_dim=self.xtrain.shape[1],activation='relu'))
        self.model.add(tf.keras.layers.Dense(64,activation='relu'))
        self.model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    def get_weights(self,population):
        total_accuracies=sum([individual[1] for individual in population])
        return [(individual[0],(individual[1]/total_accuracies)*100) for individual in population]

    def fitness_func(self,individual):
        self.create_model()  
        X_train=pd.DataFrame(data=self.scaler.fit_transform(self.xtrain),columns=self.xtrain.columns.to_list())
        X_test=pd.DataFrame(data=self.scaler.fit_transform(self.xtest),columns=self.xtest.columns.to_list())

        X_train=X_train.loc[:,[True if individual[i]=='1' else False for i in range(len(individual))]]
        X_test=X_test.loc[:,[True if individual[i]=='1' else False for i in range(len(individual))]]

        xtrain_np=np.asarray(X_train).astype(np.float64)
        xtest_np=np.asarray(X_test).astype(np.float64)
        ytrain_np=np.asarray(self.ytrain).astype(np.float64)
        ytest_np=np.asarray(self.ytest).astype(np.float64)

        print(f'{xtrain_np.shape=}  {xtest_np.shape=}  {ytrain_np.shape=}  {ytest_np.shape=}')
        self.model.fit(xtrain_np,ytrain_np,epochs=100,verbose=1)
        pred=self.model.predict(xtest_np)

        return accuracy_score(ytest_np,pred)

    def fill_population(self,individuals,goal_value=0.8):
        population=list()
        for individual in individuals:
            accuracy=self.fitness_func(individual)
            if float(accuracy)>goal_value:
                return individual,accuracy
            population.append((individual,accuracy))
        return self.get_weights(population)

    def choose_parents(self,population,counter):
        if counter==0:
            parent_1=self.get_n_individual(0,population)
            parent_2=self.get_n_individual(1,population)
        elif counter==1:
            parent_1=self.get_n_individual(0,population)
            parent_2=self.get_n_individual(2,population)
        else:
            propabilities=(individual[1] for individual in population)
            individuals=[individual[0] for individual in population]
            parent_1,parent_2=random.choices(individuals,weights=propabilities,k=2)
        
        return parent_1,parent_2

    def mutate(self,child,prob=0.1):
        new_child=copy.deepcopy(child)
        for i,charv in enumerate(new_child):
            if random.random()<prob:
                new_value='1' if charv=='0' else '0'
                new_child=new_child[:i]+new_value+new_child[i+1:]
        return new_child
    
    def reproduce(self,individual_1,individual_2):
        crosspoint=random.randint(1,len(individual_1)-1)
        child_1=individual_1[:crosspoint]+individual_2[crosspoint:]
        child_2=individual_2[:crosspoint]+individual_1[crosspoint:]
        child_1,child_2=self.mutate(child_1),self.mutate(child_2)

        return [child_1,child_2]
    
    def genaration_ahead(self,population):
        new_population=list()
        for i in range(int(len(population))//2):
            parents=self.choose_parents(population,counter=i)
            childs=self.reproduce(parents[0],parents[1])
            new_population.append(childs)
        return new_population

    def optuna_callback(self,trial):
        console=Console(record=True)
        max_iter=10000
        features=self.xtrain.shape[1]
        ind_num=trial.suggest_categorical('individual_number',[str(x * features) for x in range(2,10)])
        max_features=trial.suggest_categorical('max_features',[str(x) for x in range(self.xtrain.shape[1]//2,self.xtrain.shape[1]+1)])
        individuals=self.generate_random_individuals(int(ind_num),len(self.xtrain.columns.to_list()),int(max_features))
        population=None
        acc_score=-1
        pop_coef=self.fill_population(individuals)
        if isinstance(pop_coef,tuple):
            population,acc_score=pop_coef[0],pop_coef[1]
        new_generation=self.generation_ahead(population)
        iter_id=0
        while iter_id<max_iter:
            pop_coef=self.fill_population(new_generation)
            if isinstance(pop_coef,tuple):
                population,acc_score=pop_coef[0],pop_coef[1]
            new_generation=self.generation_ahead(population)
            console.print(f'[bold green]Iteration:{iter_id}/Max Iters:{max_iter}\tAccuracy Score:{acc_score}')
            iter_id+=1
        if population:
            GSParam.add_result(population,acc_score)
            GSParam.best_population(population,acc_score)
        return acc_score

    def solve(self,ind_num,max_features,max_iter=10):
        individuals=self.generate_random_individuals(ind_num,len(self.xtrain.columns.to_list()),max_features)
        population=self.fill_population(individuals)
        if isinstance(population,str):
            return population
        
        new_generation=self.generation_ahead(population)
        iter_id=0
        while iter_id<max_iter:
            print(f'Iteration Number:{iter_id}/Max iter:{max_iter}')
            coef=self.fill_population(new_generation)
            population=None
            accuracy=sys.maxsize
            if type(coef)==tuple:
                population,accuracy=coef[0],coef[1]
            else:
                population=coef
            if isinstance(population,str):
                break

            new_generation=self.generation_ahead(population)
            iter_id+=1
        return population

    def tune(self):
        study=optuna.create_study(direction='maximize')
        study.optimize(self.optuna_callback,n_trials=100)

        return study.best_params

class LassoSelector:
    def __init__(self,bioimage_dataframe):
        self.data=bioimage_dataframe
        self.features=self.data.columns.to_list()
        self.target=self.features[-1]
        self.features.remove(self.target)


    def optuna_callback(self,trial):
        normalize=trial.suggest_categorical('scaler',['MinMax','Standardization','KNNImpute'])
        alpha=trial.suggest_categorical('alpha',[str(x) for x in np.arange(0.5,10,0.5)])
        iterations=trial.suggest_categorical('max_iters',[str(10**x) for x in range(3,7)])
        
        scaler=MinMaxScaler() if normalize=='MinMax' else StandardScaler() if normalize=='Standardization' else KNNImputer(n_neighbors=10) 
        model=Lasso(
            alpha=float(alpha),
            tol=1e-2,
            max_iter=int(iterations),
            random_state=1234
        )

        pipe=Pipeline(
            steps=[
                ('scaler',scaler),
                ('lasso',model)
            ]
        )

        cv_model=StratifiedKFold(n_splits=10)
        X=self.data[self.features]
        Y=self.data[self.target]
        scoring=list()
        for train_indeces,test_indeces in cv_model.split(X,Y):
            xtrain=pd.DataFrame(data=[X.iloc[i].to_list() for i in train_indeces],columns=self.features)
            xtest=pd.DataFrame(data=[X.iloc[i].to_list() for i in test_indeces],columns=self.features)
            ytrain=pd.Series(data=[Y.iloc[i] for i in train_indeces],name=self.target)
            ytest=pd.Series(data=[Y.iloc[i] for i in test_indeces],name=self.target)

            pipe.fit(xtrain,ytrain)
            ypred=pipe.predict(xtest)
            scoring.append(mean_squared_error(ytest,ypred))
        
        return statistics.mean(scoring)

    def solve(self):
        study=optuna.create_study(direction='minimize')
        study.optimize(self.optuna_callback,n_trials=100)

        return study.best_params 


def scenario1():
    for dataset in os.listdir(os.path.join('..','results','extracted_features')):
        bioimage_dataframe=pd.read_csv(filepath_or_buffer=os.path.join('..','results','extracted_features',dataset),header=0)

        train_set,test_set=train_test_split(bioimage_dataframe,test_size=0.14286)
        features=bioimage_dataframe.columns.to_list()
        target=features[-1]
        features.remove(target)

        xtrain=train_set[features]
        xtest=test_set[features]
        ytrain=train_set[target]
        ytest=test_set[target]

        selector=GeneticSelector(xtrain,xtest,ytrain,ytest)
        best_parameters=selector.tune()

        with open(os.path.join('.','results','optuna','GeneticSelector',f'genetic_selector_{dataset.removesuffix(".csv")}.txt'),'w') as writer:
            writer.write('# Best parameters Genetic selector parameters')
            for param,value in best_parameters.items():
                writer.write(f'{param}\t{value}')

def scenario2():
    for dataset in os.listdir(os.path.join('..','results','extracted_features')):
        bioimage_dataframe=pd.read_csv(filepath_or_buffer=os.path.join('..','results','extracted_features',dataset),header=0)
        train_set,test_set=train_test_split(bioimage_dataframe)

        features=bioimage_dataframe.columns.to_list()
        target=features[-1]
        features.remove(target)

        xtrain=train_set[features]
        xtest=train_set[features]
        ytrain=test_set[target]
        ytest=test_set[target]

        selected_features=skgenetic_feature_selection(xtrain,xtest,ytrain,ytest)
        with open(os.path.join('.','results','optuna','GeneticSelector',f'genetic_selector_sklearn_{dataset.removesuffix(".csv")}.txt'),'w') as writer:
            writer.write(",".join(selected_features))

def scenario3():
    for dataset in os.listdir(os.path.join('..','results','extracted_features')):
        bioimage_dataframe=pd.read_csv(filepath_or_buffer=os.path.join('..','results','extracted_features',dataset),header=0)
        selector=LassoSelector(bioimage_dataframe)
        best_parameters=selector.solve()

        with open(os.path.join('.','results','optuna','LassoSelector',f'{dataset.removesuffix(".csv")}.txt'),'w') as writer:
            writer.write(f'# Best parameters Lasso selector. Extracted Parameters:{dataset.removesuffix(".csv")}')
            for param,value in best_parameters.items():
                writer.write(f'{param}\t{value}')

if __name__=='__main__':
    # scenario1() # 1. Custom feature selector using genetic algorithm
    # scenario2() # 2. sklearn genetic selector
    scenario3() # 3. Feature selection using Lasso Feature Selection