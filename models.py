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
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

import tensorflow as tf
import tensorflow_addons as tfa


from dataset import Dataset,PatchDataset
from rich.console import Console
from functools import reduce


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

    # Feature extraction function
    def extract_features(self, base_model_name, lb=0, ub=0, save=True, save_best=False):
        self.pretrained_model_name = base_model_name
        bioimage_input, bioimage_label = self.dataset.load_image_dataset(split=False)
        self.feature_extraction_model(base_model_name, lb, ub)
        self.console.print(f'[bold green] {self.model.summary()}')

        feature_set = list()
        labels = list()
        for i in range(0, bioimage_input.shape[0] // Dataset.batch_size):
            batch_indeces = list(range(i, i + Dataset.batch_size))
            batch_images = []

            for image_index in batch_indeces:
                image = np.array(bioimage_input[image_index])
                image = np.expand_dims(image, axis=0)
                image = tf.keras.applications.imagenet_utils.preprocess_input(image)
                batch_images.append(image)

            batch_images = np.vstack(batch_images)
            features = self.model.predict(batch_images, batch_size=Dataset.batch_size)
            output_shape = self.model.layers[-1].output.shape
            flatten_shape = reduce(lambda a, b: a * b, [x for x in list(output_shape) if type(x) == int])
            features = np.reshape(features, (-1, flatten_shape))
            feature_set.append(features)
            labels.append(np.array(bioimage_label[i:i + Dataset.batch_size]))

        # Convert features and labels to arrays
        feature_set = np.vstack(feature_set)
        labels = np.hstack(labels)

        # Save extracted features and labels to CSV file
        if save:
            with open(os.path.join('', 'project_results', 'Features', f'ExpanderNet_{base_model_name}_{lb}_{ub}_feature_set_{datetime.now().strftime("%m%d%Y%H%M%S")}.csv'), 'w') as f:
                columns = None
                for i in range(feature_set.shape[0]):
                    for j in range(feature_set.shape[1]):
                        if columns is None:
                            columns = feature_set.shape[1]
                        f.write(f'{feature_set[i][j]},')
                    f.write(f'{labels[i]}\n')
        
        print(feature_set,end='\n\n')
        print(labels)


    # Models that potentially could be use in an experiment
    def feature_extraction_model(self,base_model_name='vgg16',optimizer='adam',selective_fine_tuning=True,lb=0,ub=0):
        """
        Creates a feature extraction model in order to extract features from the bioimage dataset
        Parameters
        ----------
        base_model_name: str
                Pretrained model to use, possible values[vgg16,vgg19,ResNet50,ResNet101]
                It is possible versions 2 for some models to be used. Check the compatibility
        optimizet: tf.keras.optimizers.Optimizer
            Select the optimization sequence in the feature extraction pretrained model
        selective_fine_tuning: bool
        lb: int
            selective fine tuning lower bound(most likely 0)
        ub: int
            selective fine tuning upper bound(differs based onm each pretrained model)
        
        Returns: tf.keras.applications.Model
            based opn the user selection a model architecture  is returned in order be used from the bioimages data
        """
        base_model,self.pretrained_model_name=(tf.keras.applications.VGG16(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG16') if base_model_name=='vgg16' else (tf.keras.applications.VGG19(weighqts='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG19') if base_model_name=='vgg19' else (tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET50') if base_model_name=='resnet50' else (tf.keras.applications.ResNet101V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET101')
        base_model.trainable = True
        if selective_fine_tuning:
            for layer in base_model.layers[lb:ub]:
                layer.trainable = False
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=Dataset.image_size,name='feature_extractor_input_layer'))
        self.model.add(base_model)
        self.model.add(tf.keras.layers.GlobalAveragePooling2D(name='average_pooling_layer'))
    
    def premade_model(self,base_model_name='vgg16',optimizer_name='adam',selective_fine_tuning=(None,-1,-1)):
        base_model,self.pretrained_model_name=(tf.keras.applications.VGG16(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG16') if base_model_name=='vgg16' else (tf.keras.applications.VGG19(weighqts='imagenet', input_shape=Dataset.image_size, include_top=False),'VGG19') if base_model_name=='vgg19' else (tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET50') if base_model_name=='resnet50' else (tf.keras.applications.ResNet101V2(weights='imagenet', input_shape=Dataset.image_size, include_top=False),'RESNET101')
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


    def conventional_model(self,xtrain,xtest,ytrain,scaling="MinMax",feature_extraction="FFS",clf="ada"):
        # 1. Scaling features
        scaler= MinMaxScaler() if scaling == 'MinMax' else StandardScaler() if scaling == 'Standardization' else KNNImputer()
        
        # 2. Feature selection
        lasso_pipeline = Pipeline([
            ('scaler', scaler),
            ('lasso', Lasso())
        ])

        param_grid={"lasso_alpha":[0.1,0.5,1.0,1.5,2.0,5.0]}
        scorer=make_scorer(r2_score)

        lasso_grid = GridSearchCV(lasso_pipeline, param_grid=param_grid, cv=5,scoring=scorer,n_jobs=psutil.cpu_count(logical=False))
        lasso_grid.fit(xtrain,xtrain)
        coefficients=lasso_grid.best_estimator_.coef_
        feature_importance=np.abs(coefficients)

        xtrain, xtest = pd.DataFrame(xtrain), pd.DataFrame(xtest)
        xtrain.drop([column_name for i, column_name in enumerate(xtrain.columns.tolist()) if feature_importance[i]==0], axis=1, inplace=True)
        xtest.drop([column_name for i, column_name in enumerate(xtest.columns.tolist()) if feature_importance[i]==0], axis=1, inplace=True)

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
    
    def voting_model(self,xtrain,xtest,ytrain,scaling="MinMax",feature_extraction="FFS",voting='soft'):
        # 1. Scaling features
        scaler= MinMaxScaler() if scaling == 'MinMax' else StandardScaler() if scaling == 'Standardization' else KNNImputer()
        
        # 2. Feature selection
        lasso_pipeline = Pipeline([
            ('scaler', scaler),
            ('lasso', Lasso())
        ])
        param_grid={"lasso_alpha":[0.1,0.5,1.0,1.5,2.0,5.0]}
        scorer=make_scorer(r2_score)
        lasso_grid = GridSearchCV(lasso_pipeline, param_grid=param_grid, cv=5,scoring=scorer,n_jobs=psutil.cpu_count(logical=False))
        lasso_grid.fit(xtrain,xtrain)
        coefficients=lasso_grid.best_estimator_.coef_
        feature_importance=np.abs(coefficients)
        xtrain, xtest = pd.DataFrame(xtrain), pd.DataFrame(xtest)
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
    
    # Optuna callback 
    def optuna_callback(self,trial):
        selective_fine_tuning_base_model = trial.suggest_categorical('selective_fine_tuning_base_model',['vgg16','vgg19','resnet50','resnet101'])
        selective_fine_tuning_ub = trial.suggest_categorical('selective fine tuning',['0','1','2'])
        optimizer=trial.suggest_categorical('optimizer',['adam','sgd'])
        scaling = trial.suggest_categorical('scaling',['MinMax','Standardization','KnnImputer'])
        classifier = trial.suggest_categorical('classifier',Controller.classifiers)
        if classifier=='vt':
            voting_system=trial.suggest_categorical('voting',['soft','hard'])
        self.clear_session()

        feature_map,label_map=self.dataset.load_image_dataset(split=False)
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

    def predict(self,xset):
        if not hasattr(self.model,"predict"):
            raise AttributeError(f"No method predict in {type(self.model)}")
        return self.model.predict(xset)

    def graph(self):
        plt.figure(figsize=(15,8))
        plt.title(f"Transfer model - {self.base_model_name} - Architecture Diagram")
        tf.keras.utils.plot_model(model=self, to_file=os.path.join('','project_results','figures','history','_graph.png'), show_shapes=True, show_layer_names=True)
        plt.imshow(imread(os.path.join('','project_results','figures','history','ExpanderNet_feature_extraction.png')))
        plt.xticks([]), plt.yticks([])
        plt.show()
