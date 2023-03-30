import os,json,random
import tensorflow as tf,numpy as np
from rich.console import Console

class JSON_Parser:
    path_to_data_files = os.path.join('..','Dataset')

    def __init__(self):
        self._training_json_file = os.path.join(JSON_Parser._path_to_data_files, 'Extracted Annotation Data', 'training_set.json')
        self._validation_json_file = os.path.join(JSON_Parser._path_to_data_files, 'Extracted Annotation Data', 'validation_set.json')

    def load_json_files(self):
        with open(self._training_json_file, 'r') as training_data, open(self._validation_json_file, 'r') as validation_data:
                self._trainData = json.load(training_data)
                self._validationData = json.load(validation_data)
        return (self._trainData, self._validationData)

    def concatenate_json_data(self):
        trainData, validationData = self.load_json_files()
        self.allData = dict()
        self.allData = trainData | validationData
        return list(self.allData.values())


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