import os,numpy as np,psutil,seaborn as sns,logging,cv2,pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score,classification_report,log_loss,r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from collections import defaultdict

import tensorflow as tf
import tensorflow_addons as tfa
import pickle,oapackage,statistics,optuna


from dataset import Dataset,PatchDataset
from feature_selection import LassoSelector
from rich.console import Console
from rich.table import Table
from functools import reduce
from elayers import AddLayer
from feature_selection import GeneticSelector,LassoSelector

class Controller:
    freezing_layers = {
        'vgg16': [6,10,14],
        'vgg19': [6,11,16],
        'resnet50': [39,85,153],
        'resnet101': [39,85,340]
    }

    estimators={
                'dt': DecisionTreeClassifier(max_depth=8,random_state=1234),
                'rf': RandomForestClassifier(n_estimators=100,random_state=1234,max_depth=8),
                'adaboost': AdaBoostClassifier(learning_rate=1e-5),
                'knn':KNeighborsClassifier(n_neighbors=10,p=2)
    }

    classifiers = ['naive_bayes','svm','rf','knn','adaboost']

class ImageNet:
    path_to_furhman1 = os.path.join('..','dataset','40x','Fuhrman 1')
    path_to_furhman2 = os.path.join('..','dataset','40x','Fuhrman 2')
    path_to_furhman3 = os.path.join('..','dataset','40x','Fuhrman 3')

    @staticmethod
    def augmentation_model():
        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(mode='horizontal_and_vertical', name='random_flip'),
                tf.keras.layers.RandomRotation(factor=0.9, fill_mode='nearest', name='random_rotation'),
                tf.keras.layers.Rescaling(1./255, name='normalization')
            ],
            name='augmentation_model'
        )

    @staticmethod
    def optimizer(opt_name):
        return tf.keras.optimizers.Adam(learning_rate=1e-5) if opt_name=='adam' else tf.keras.optimizers.SGD(learning_rate=1e-5) if opt_name=='sgd' else tf.keras.optimizers.RMSprop(learning_rate=1e-5) if opt_name=='rmsprop' else tf.keras.optimizers.Adagrad(learning_rate=1e-5)  

    
    def __init__(self):
        self.dataset=Dataset()
        self.model=None
        self.pretrained_model_name=''
        self.log=None
        self.console=None
        self.bioimage_dataframe=None
        self.bioimage_input, self.bioimage_label = self.dataset.load_image_dataset(split=False)
        self.initialize_tui()

    def initialize_tui(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(filename=os.path.join('','project_results',''))
        formatter=logging.Formatter('%(asctime)s\t%(message)s')
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.log=logging.getLogger(name=f'rcc_staging_logger')
        self.log.addHandler(sh)
        self.log.addHandler(fh)
        self.console=Console(record=True)
    
    def clear_session(self):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        tf.keras.backend.clear_session()
        del self.model

    def plot_random_samples(self):
        plt.figure(figsize=(12,10))

    # Feature extraction function
    def extract_features(self, base_model_name, lb=0, ub=0, save=False):
        self.pretrained_model_name = base_model_name
        self.feature_extraction_model(base_model_name, lb, ub)
        self.console.print(f'[bold green] {self.model.summary()}')

        feature_set = list()
        labels = list()
        for i in range(0, self.bioimage_input.shape[0] // Dataset.batch_size):
            batch_indeces = list(range(i, i + Dataset.batch_size))
            batch_images = []

            for image_index in batch_indeces:
                image = np.array(self.bioimage_input[image_index])
                image = np.expand_dims(image, axis=0)
                image = tf.keras.applications.imagenet_utils.preprocess_input(image)
                batch_images.append(image)

            batch_images = np.vstack(batch_images)
            features = self.model.predict(batch_images, batch_size=Dataset.batch_size)
            output_shape = self.model.layers[-1].output.shape
            flatten_shape = reduce(lambda a, b: a * b, [x for x in list(output_shape) if type(x) == int])
            features = np.reshape(features, (-1, flatten_shape))
            feature_set.append(features)
            labels.append(np.array(self.bioimage_label[i:i + Dataset.batch_size]))

        # Convert features and labels to arrays
        feature_set = np.vstack(feature_set)
        labels = np.hstack(labels)

        # Save extracted features and labels to CSV file
        data=list()
        # Save extracted features and labels to CSV file
        if save:
            with open(os.path.join('', 'results', 'extracted_features', f'ImageNet_patches_{base_model_name}_{lb}_{ub}.csv'), 'w') as writer:
                for i in range(feature_set.shape[1]):
                    writer.write(f'F{i+1},')
                writer.write('Fuhrman\n')
                for i in range(feature_set.shape[0]):
                    row=list()
                    for j in range(feature_set.shape[1]):
                        row.append(feature_set[i][j])
                        writer.write(f'{feature_set[i][j]},')
                    writer.write(f'{labels[i]}\n')
                    row.append(labels[i])
                    data.append(row)
        else:
            for i in range(feature_set.shape[0]):
                row=list()
                for j in range(feature_set.shape[1]):
                    row.append(feature_set[i][j])
                row.append(labels[i])
                data.append(row)
        self.bioimage_dataframe=pd.DataFrame(data,columns=[f'F{i+1}' for i in range(len(feature_set[0]))]+['Fuhrman'])


    # Models that potentially could be use in an experiment
    def feature_extraction_model(self,base_model_name='vgg16',optimizer='adam',selective_fine_tuning=True,lb=0,ub=0):
        """
        Creates a feature extraction model in order to extract features from the bioimage dataset
        Parameters
        ----------
            * base_model_name: str
                    Pretrained model to use, possible values[vgg16,vgg19,ResNet50,ResNet101]
                    It is possible versions 2 for some models to be used. Check the compatibility
            * optimizer: tf.keras.optimizers.Optimizer
                Select the optimization sequence in the feature extraction pretrained model
            * selective_fine_tuning: bool
            * lb: int
                selective fine tuning lower bound(most likely 0)
            * ub: int
                selective fine tuning upper bound(differs based onm each pretrained model)
            
            * Returns: tf.keras.applications.Model
                based opn the user selection a model architecture  is returned in order be used from the bioimages data
        """
        base_model,self.pretrained_model_name=(tf.keras.applications.VGG16(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG16') if base_model_name=='vgg16' else (tf.keras.applications.VGG19(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG19') if base_model_name=='vgg19' else (tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET50') if base_model_name=='resnet50' else (tf.keras.applications.ResNet101V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET101')
        base_model.trainable = True
        if selective_fine_tuning:
            for layer in base_model.layers[lb:ub]:
                layer.trainable = False
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=Dataset.image_size,name='feature_extractor_input_layer'))
        self.model.add(base_model)
        self.model.add(tf.keras.layers.GlobalAveragePooling2D(name='average_pooling_layer'))
        print(self.model.summary())
    
    def premade_model(self,base_model_name='vgg16',optimizer_name='adam',selective_fine_tuning=(None,-1,-1)):
        base_model,self.pretrained_model_name=(tf.keras.applications.VGG16(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG16') if base_model_name=='vgg16' else (tf.keras.applications.VGG19(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG19') if base_model_name=='vgg19' else (tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET50') if base_model_name=='resnet50' else (tf.keras.applications.ResNet101V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET101')
        base_model.trainable = True
        if selective_fine_tuning[0]:
            for layer in base_model.layers[selective_fine_tuning[1]:selective_fine_tuning[2]]:
                layer.trainable = False

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=Dataset.image_size,name='feature_extractor_input_layer'))
        self.model.add(ImageNet.augmentation_model())
        self.model.add(base_model)
        self.model.add(tf.keras.layers.GlobalAveragePooling2D(name='average_pooling_layer'))
        self.model.add(tf.keras.layers.Dense(units=self.dataset.num_classes,activation='softmax',name='activation_layer'))
        self.model.compile(ImageNet.optimizer(optimizer_name),loss='sparse_categorical_crossentropy', metrics=['accuracy', tfa.metrics.CohenKappa(num_classes=self.dataset.num_classes,sparse_labels=True), tfa.metrics.F1Score(num_classes=self.dataset.num_classes, average='macro')])

        self.model.save(os.path.join('..','fine_tuned_models',f'fined_tuned_{base_model_name}.h5'))


    def conventional_model(self,xtrain,xtest,ytrain,ytest,scaling="MinMax",feature_selection="lasso",clf="ada"):
        # # 1. Scaling features
        # scaler= MinMaxScaler() if scaling == 'MinMax' else StandardScaler() if scaling == 'Standardization' else KNNImputer()
        
        # # 2. Feature selection
        # if feature_selection=='lasso':
        #     lasso_pipeline = Pipeline(
        #     steps=[
        #         ('scaler', scaler),
        #         ('lasso', Lasso())
        #     ],verbose=True)

        #     param_grid={"lasso__alpha":list(range(10,20))}
        #     scorer=make_scorer(r2_score)

        #     lasso_grid = GridSearchCV(lasso_pipeline, param_grid=param_grid, cv=5,scoring=scorer,n_jobs=psutil.cpu_count(logical=False))
        #     lasso_grid.fit(xtrain,xtrain)
        #     coefficients=lasso_grid.best_estimator_.coef_
        #     feature_importance=np.abs(coefficients)
        #     xtrain.drop([column_name for i, column_name in enumerate(xtrain.columns.tolist()) if feature_importance[i]==0], axis=1, inplace=True)
        #     xtest.drop([column_name for i, column_name in enumerate(xtest.columns.tolist()) if feature_importance[i]==0], axis=1, inplace=True)

        selector=LassoSelector(xtrain,xtest,ytrain,ytest)
        params=selector.solve()
        print(params)
        
        # elif feature_selection=='ga':
        #     selector=GeneticSelector(xtrain,xtest,ytrain,ytest)
        #     selector.solve()


        # 3. classification model
        del self.model
        self.model = None
        if clf == 'naive_bayes':
            self.model = GaussianNB()
        elif clf == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=10,metric='euclidean')
        elif clf == 'svm':
            self.model = SVC(kernel='rbf',decision_function_shape='ovo', break_ties=False)
        elif clf == 'rf':
            self.model = RandomForestClassifier(n_estimators=100,max_depth=8,class_weight='balanced',random_state=1234)
        elif clf == 'adaboost':
            self.model = AdaBoostClassifier(learning_rate=1e-5)
        
        self.model=self.model.fit(xtrain,ytrain)
    
    def voting_model(self,xtrain,xtest,ytrain,scaling="MinMax",feature_selection="lasso",voting='soft'):
        # 1. Scaling features
        scaler= MinMaxScaler() if scaling == 'MinMax' else StandardScaler() if scaling == 'Standardization' else KNNImputer()
        
        # 2. Feature selection
        if feature_selection=='lasso':
            lasso_pipeline = Pipeline(
                steps=[
                    ('scaler', scaler),
                    ('lasso', Lasso(max_iter=100000))
                ],verbose=True
            )
            param_grid={"lasso__alpha":[0.1,0.5,1.0,1.5,2.0,4.0,5.0]}
            scorer=make_scorer(r2_score)
            lasso_grid = GridSearchCV(lasso_pipeline, param_grid=param_grid, cv=5,scoring=scorer,n_jobs=psutil.cpu_count(logical=False))
            lasso_grid.fit(xtrain,xtrain)
            coefficients=lasso_grid.best_estimator_.coef_
            feature_importance=np.abs(coefficients)
            xtrain.drop([column_name for i, column_name in enumerate(xtrain.columns.tolist()) if feature_importance[i]==0], axis=1, inplace=True)
            xtest.drop([column_name for i, column_name in enumerate(xtest.columns.tolist()) if feature_importance[i]==0], axis=1, inplace=True)
        
        del self.model
        voting_clf = VotingClassifier(
            estimators = [
                GaussianNB(),
                KNeighborsClassifier(n_neighbors=10),
                DecisionTreeClassifier(max_depth=8,class_weight='balanced'),
                SVC(kernel='rbf',decision_function_shape='ovo',break_ties=True),
                RandomForestClassifier(max_depth=8,n_estimators=50,verbose=True),                
                AdaBoostClassifier(n_estimators=50,learning_rate=1e-4)                
            ],
            voting = voting,
            flatten_transform = False,
            verbose = True
        )
        self.model = voting_clf.fit(xtrain,ytrain)
    
    def custom_model(self,optimizer_name):
        self.clear_session()
        del self.model

        self.model=tf.keras.Sequential(
            layers=[
                tf.keras.layers.Input(shape=Dataset.image_size,name='input_layer'),
                ImageNet.augmentation_model(),
                tf.keras.layers.Conv2D(filters=32,kernel_size=(1,1),strides=(1,1),padding='valid',kernel_regularizer=tf.keras.regularizers.l2(1e-2),name='conv1'),
                tf.keras.layers.Activation(activation='relu',name='relu_1'),
                tf.keras.layers.BatchNormalization(name='batch_norm_1'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='valid',name='max_pool_1'),

                tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-2), name='conv_2'),
                tf.keras.layers.Activation(activation='relu', name='relu_2'),
                tf.keras.layers.BatchNormalization(name='batch_norm_2'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='valid', name='max_pool_2'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-2), name='conv_3'),
                tf.keras.layers.Activation(activation='relu', name='relu_3'),
                tf.keras.layers.BatchNormalization(name='batch_norm_3'),
                tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='valid', name='max_pool_3'),

                AddLayer(),

                tf.keras.layers.Flatten(name='flatten'),
                tf.keras.layers.Dense(units=2048, activation='relu', name='dense_1'),
                tf.keras.layers.Dense(units=self.dataset.num_classes, activation='softmax', name='softmax')
            ],name='RCC_resnet'
        )
        self.model.compile(ImageNet.optimizer(optimizer_name),loss='sparse_categorical_crossentropy',metrics=['accuracy', tfa.metrics.CohenKappa(num_classes=self.dataset.num_classes,sparse_labels=True), tfa.metrics.F1Score(num_classes=self.dataset.num_classes, average='micro')])

    # Optuna callback 
    def optuna_callback(self,trial):
        selective_fine_tuning_base_model = trial.suggest_categorical('selective_fine_tuning_base_model',['vgg16','vgg19','resnet50','resnet101'])
        selective_fine_tuning_ub = trial.suggest_categorical('selective fine tuning',['0','1','2'])
        scaling = trial.suggest_categorical('scaling',['MinMax','Standardization','KnnImputer'])
        classifier = trial.suggest_categorical('classifier',Controller.classifiers)
        if classifier=='vt':
            voting_system=trial.suggest_categorical('voting',['soft','hard'])
        self.clear_session()

        self.extract_features(selective_fine_tuning_base_model,ub=selective_fine_tuning_ub,save=False)

        feature_names=self.bioimage_dataframe.columns.to_list()
        target_name=feature_names[-1]
        feature_names.remove(target_name)

        feature_map,label_map=self.bioimage_dataframe[feature_names],self.bioimage_dataframe[target_name]
        scores = dict(accuracy=list(),recall=list(),precision=list(),f1_score=list(),cohens_kappa=list())
        cv_model = StratifiedKFold(n_splits=10,random_state=1234,shuffle=True)
        for train_indeces,test_indeces in cv_model.split(feature_map,label_map):
            xtrain = pd.DataFrame(data=[feature_map.iloc[i].tolist() for i in train_indeces],columns=feature_map.columns)
            xtest = pd.DataFrame(data=[feature_map.iloc[i].tolist() for i in test_indeces],columns=feature_map.columns)
            ytrain = pd.Series(data=[label_map.iloc[i] for i in train_indeces],name=label_map.name)
            ytest = pd.Series(data=[label_map.iloc[i] for i in test_indeces],name=label_map.name)
            if classifier=='vt':
                self.voting_model(xtrain,xtest,ytrain,scaling=scaling,voting=voting_system)
            else:
                self.conventional_model(xtrain,xtest,ytrain,scaling=scaling,clf=classifier)
            
            ytest=np.array(ytest)
            predictions = np.array(self.model.predict(xtest))

            acc_score,recall,precision,f1score,ck=accuracy_score(ytest,predictions),recall_score(ytest,predictions),precision_score(ytest,predictions),f1_score(ytest,predictions,average='macro'),cohen_kappa_score(ytest,predictions)
            scores['accuracy'].append(acc_score)
            scores['recall'].append(recall)
            scores['precision'].append(precision)
            scores['f1_score'].append(f1score)
            scores['cohens_kappa'].append(ck)            

            self.console(f'[bold red]:right arrow: [bold green]Accuracy:{acc_score}')
            self.console(f'[bold red]:right arrow: [bold green]Recall Score:{recall}')
            self.console(f'[bold red]:right arrow: [bold green]Precision Score:{precision}')
            self.console(f'[bold red]:right arrow: [bold green]F1 Score:{f1score}')
            self.console(f'[bold red]:right arrow: [bold green]Cohens Kappa:{ck}')
            print(end='\n\n')
    
    # -->Fit ImageNet model in to a dataset
    def fit(self,xtrain,ytrain):
        self.log.debug(self.model)
        if not hasattr(self.model,"fit"):
            raise AttributeError(f"No method fit found in {type(self.model)}")
        
        if type(self.model)==tf.keras.Sequential:
            self.model.fit(
                xtrain,
                ytrain,
                callbacks=ImageNet.callbacks(self.pretrained_model_name),
                batch_size=Dataset.batch_size,
                steps_per_epoch=xtrain.shape[0]//Dataset.batch_size,
                verbose=True
            )
        else:
            self.model.fit(xtrain,ytrain) 

    def predict(self,xtest):
        if not hasattr(self.model,"predict"):
            raise AttributeError(f"No method predict in {type(self.model)}")
        return self.model.predict(xtest)

    # Statistical representation of the data and model summary
    def graph(self):
        plt.figure(figsize=(15,8))
        plt.title(f"Transfer model - {self.base_model_name} - Architecture Diagram")
        tf.keras.utils.plot_model(model=self, to_file=os.path.join('','project_results','figures','history','_graph.png'), show_shapes=True, show_layer_names=True)
        plt.imshow(imread(os.path.join('','project_results','figures','history','ExpanderNet_feature_extraction.png')))
        plt.xticks([]), plt.yticks([])
        plt.show()

    def statistics(self):
        stats_table=Table(headers=['','Mean','Median','Std','IQR','Skewness','Kurtosis','Entropy'],header_style='bold')
        feature_stats=defaultdict(dict)
        for column in self.bioimage_dataframe.columns.to_list():
            data=self.bioimage_dataframe[column].to_list()
            feature_stats[column]['mean']=statistics.mean(data)
            feature_stats[column]['std']=statistics.stdev(data)
            feature_stats[column]['median']=statistics.median(data)
            feature_stats[column]['Skewness']=statistics.skew(data)
            q1,q3=statistics.quantiles(data)
            feature_stats[column]['iqr']=q3-q1
            feature_stats[column]['kurtosis']=statistics.kurtosis(data)
            feature_stats[column]['Entropy']=Dataset.entropy(data)
            row=[column]
            row.extend(feature_stats[column].values())
            stats_table.add_row(",".join(row),style='green')

        self.console.rule('[bold red]Biomadical images dataset descriptive analytics')
        self.console.print(stats_table)


class OptunaModel:
    study_case_results_path=os.path.join('','optuna')
    study_results=dict()

    @staticmethod
    def flush():
        OptunaModel.results.clear()

    @staticmethod
    def add_result(combination,metric_name,metric_value):
        if OptunaModel.study_results.get(combination,None)==None:
            OptunaModel[combination]=dict()
        OptunaModel[combination][metric_name]=metric_value

    def __init__(self,new_study_case):
        self.study_id=new_study_case
        self.study = None
        self.trial = None
        self.study_objective_dimensions=list()

    def add_objective_dimension(self,param_name,direction):
        self.study_objective_dimensions.append((param_name,direction))
    
    def set_callback(self,callback_function):
        del self.study
        del self.trial
        if self.study_objective_dimensions==[]:
            raise AttributeError("No objective dimension has been setted properly")

        self.study=optuna.create_study(directions=[direction for _,direction in self.study_objective_dimensions],load_if_exists=True)
        self.study.optimize(callback_function,n_trials=50)
    
    def export(self):
        self.study.save_study_direction()
        self.study.trials_dataframe().to_csv(os.path.join(OptunaModel.study_case_results_path,f'{self.study_id}.csv'))

    def export_best(self):
        if not os.path.exists(os.path.join(OptunaModel.study_case_results_path,self.study_id)):
            os.mkdir(os.path.join(OptunaModel.study_case_results_path,self.study_id))
        
        with open(os.path.join(OptunaModel.study_case_results_path,self.study_id,'best_trial.pkl'), 'wb') as f:
            pickle.dump(self.study.best_trial, f)

        # save the best parameters
        with open(os.path.join(OptunaModel.study_case_results_path,self.study_id,'best_params.pkl'), 'wb') as f:
            pickle.dump(self.study.best_params, f)

        # save the best value of the objective function
        with open(os.path.join(OptunaModel.study_case_results_path,self.study_id,'best_value.pkl'), 'wb') as f:
            pickle.dump(self.study.best_value, f)
    
    def pareto_front(self):
        metrics=[
            'accuracy',
            'f1-score',
            'cohens_kappa'
        ]

        pareto_objects=oapackage.ParetoDoubleLong()