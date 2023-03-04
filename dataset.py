import tensorflow as tf
import matplotlib.pyplot as plt
from math import sqrt
import os,cv2
from rich.console import Console
from collections import Counter
import numpy as np
import optuna,pickle

from im_parser import JSON_Parser


class Dataset:
    path_to_bioimages=os.path.join('dataset','40x')
    path_to_patches=os.path.join('dataset','Extracted Annotation Data','Cleared Image Patches')
    test_size=0.14286
    image_size=(64,64,3)
    batch_size=16
    
    @staticmethod
    def change_patches_path(new_path_to_patches):
        Dataset.path_to_patches=new_path_to_patches
    
    @staticmethod
    def change_bioimages_path(new_images_path):
        Dataset.path_to_bioimages=new_images_path
    
    def __init__(self):
        self.class_names=os.listdir(Dataset.path_to_bioimages)
        self.num_classes=len(self.class_names)
    
    def load_image_dataset(self,split=False):
        if not split:
            images = tf.keras.utils.image_dataset_from_directory(            
                directory=Dataset.path_to_patches,
                validation_split=None,
                subset=None,
                seed=1234,
                batch_size=Dataset.batch_size,
                color_mode='rgb',
                image_size=Dataset.image_size[:2],
                interpolation='lanczos5',
                crop_to_aspect_ratio=False
            )

            raw_images,labels=tuple(zip(*images))
            return tf.concat([image for image in raw_images],0),tf.concat([label for label in labels],0)

        else:
            training_set=tf.keras.utils.image_dataset_from_directory(
                directory=Dataset.path_to_patches,
                validation_split=0.14286,
                subset='training',
                seed=89,
                batch_size=Dataset.batch_size,
                color_mode='rgb',
                image_size=Dataset.image_size[:2], 
                interpolation='lanczos5',
                crop_to_aspect_ratio=False,
                shuffle=True
            )

            test_set=tf.keras.utils.image_dataset_from_directory(
                directory=Dataset.path_to_patches,
                validation_split=0.14286,
                subset='validation',
                seed=89,
                batch_size=Dataset.batch_size,
                color_mode='rgb',
                image_size=Dataset.image_size[:2], 
                interpolation='lanczos5',
                crop_to_aspect_ratio=False
            )

            self.num_classes = len(training_set.class_names)
            self.class_names = test_set.class_names
            train_images, train_labels = tuple(zip(*training_set))
            test_images, test_labels = tuple(zip(*test_set))
            return tf.concat([image for image in train_images], 0), tf.concat([label for label in train_labels], 0), tf.concat([image for image in test_images], 0), tf.concat([label for label in test_labels], 0)
        
    def plot(self,image_set:tf.Tensor):
        plt.figure(figsize=(12,8))
        plt.suptitle("Loaded Dataset - Random Samples")
        for images, labels in image_set.take(1):
            for i in range(Dataset.batch_size):
                plt.subplot(int(sqrt(Dataset.batch_size)), int(sqrt(Dataset.batch_size)), i+1)
                # Pixel values normalization for removing the negative values
                plt.imshow(tf.cast(images[i]/255, tf.float32))
                plt.title(Dataset.class_names[labels[i]])
                plt.axis('off')
    
    def statistics(self):
        console=Console(record=True)
        console.rule('[bold red]Biomadical images dataset descriptive analytics')
    # TO BE FILLED

class PatchDataset:
    def __init__(self, imageNum:int, annotData:list):
        self.image_root_folder = os.path.join(JSON_Parser._path_to_data_files, '40x')
        self.annotData = sorted(annotData, key=lambda x: x['image_attributes']['name'])
        self.imageNum = imageNum
        
    def check_image_files(self)->list:
        self.image_file_paths = []
        for root, dirs, files in os.walk(self._image_root_folder):
            for file in files:
                self.image_file_paths.append(os.path.abspath(os.path.join(root, file)))
        
        # Sorting the paths by filename.jpg
        norm_paths = []
        for path in self.image_file_paths:
            norm_path = os.path.normpath(path)
            norm_path = norm_path.split(os.sep)
            norm_paths.append(norm_path)
        norm_paths = sorted(norm_paths, key=lambda x: x[-1][:-4])
        
        for i in range(len(norm_paths)):
            norm_paths[i] = '/'.join(norm_paths[i])
        
        self.image_file_paths = norm_paths
        return self.image_file_paths
    
    @property
    def image_info(self):
        self.paths = self.check_image_files()
        self.imageName = self.annotData[self.imageNum]['image_attributes']['name']
        self.imageProps = self.annotData[self.imageNum]['regions']
        return (self.paths, self.imageName, self.imageProps)
    
    def extract_image_patches(self, extract_masks=False, extract_image_patches=False, display_annotations=False):
        self.paths, self.imageName, self.imageProps = self.image_info
        
        for i in range(len(self.image_file_paths)):
            if self.imageName in self.paths[i]:
                img = cv2.imread(self.paths[i])
                imgBW = np.zeros_like(img)
                imgBW_filled = imgBW.copy()
                
                # Step 1: Extracting annotation masks, annotation labels and coordinates
                gatheredAnnot_info = []
                for j in range(len(self.imageProps)):
                    coords_zipped = zip(self.imageProps[j]['shape_attributes']['all_points_x'],
                                        self.imageProps[j]['shape_attributes']['all_points_y'])
                    annotCoords = np.array([list(x) for x in coords_zipped])
                    imgBW_filled = cv2.bitwise_or(imgBW_filled, cv2.fillPoly(imgBW, pts=np.int32(np.ceil([annotCoords])), color=(255,255,255)))
                    gatheredAnnot_info.append([self.imageProps[j]['region_attributes']['name'].title(), np.int32(np.ceil(annotCoords))])
                
                if extract_masks==True:
                    gatheredAnnot_info_array = np.array(gatheredAnnot_info)
                    c = Counter(gatheredAnnot_info_array[:,0])
                    majority_annotLabel = c.most_common()[0][0]
                    if majority_annotLabel.title() == 'Fuhrman 1':
                        cv2.imwrite('../dataset/Extracted Annotation Data/Annotation Masks/Fuhrman 1/' + self.imageName[:-4] + '.jpg', imgBW_filled)
                        txt_path = f'../dataset/Extracted Annotation Data/Annotation Masks/Fuhrman 1/{self.imageName[:-4]}.txt'
                        with open (txt_path, 'w') as f:
                            np.savetxt(f, gatheredAnnot_info, fmt='%s', delimiter=', ')
                    elif majority_annotLabel.title() == 'Fuhrman 2':
                        cv2.imwrite('../dataset/Extracted Annotation Data/Annotation Masks/Fuhrman 2/' + self.imageName[:-4] + '.jpg', imgBW_filled)
                        txt_path = f'../dataset/Extracted Annotation Data/Annotation Masks/Fuhrman 2/{self.imageName[:-4]}.txt'
                        with open (txt_path, 'w') as f:
                            np.savetxt(f, gatheredAnnot_info, fmt='%s', delimiter=', ')
                    elif majority_annotLabel.title() == 'Fuhrman 3':
                        cv2.imwrite('../dataset/Extracted Annotation Data/Annotation Masks/Fuhrman 3/' + self.imageName[:-4] + '.jpg', imgBW_filled)
                        txt_path = f'../dataset/Extracted Annotation Data/Annotation Masks/Fuhrman 3/{self.imageName[:-4]}.txt'
                        with open (txt_path, 'w') as f:
                            np.savetxt(f, gatheredAnnot_info, fmt='%s', delimiter=', ')
                                
                # Step 2: Extracting the image patches
                imgBW = cv2.cvtColor(imgBW_filled.copy(), cv2.COLOR_BGR2GRAY)
                imgBW = cv2.threshold(imgBW, 127, 255, cv2.THRESH_BINARY)[1]
                                
                # Clearing the image background for extracting clear image patches
                imgCleared = img.copy()
                imgCleared[imgBW==0] = 0
                
                counter_1 = counter_2 = counter_3 = 0
                contours = cv2.findContours(imgBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                
                if extract_image_patches==True:
                    for k, cntr in zip(range(len(self.imageProps)), contours):
                        x, y, w, h = cv2.boundingRect(cntr)
                        if (self.imageProps[k]['region_attributes']['name'].title() == 'Fuhrman 1'):
                            counter_1 += 1
                            imgCrop = img[y:y+h, x:x+w]
                            cv2.imwrite(f'../dataset/Extracted Annotation Data/Image Patches/Fuhrman 1/{self.imageName[:-4]}_{counter_1}.jpg', imgCrop)
                            imgCrop_clear = imgCleared[y:y+h, x:x+w]
                            cv2.imwrite(f'../dataset/Extracted Annotation Data/Cleared Image Patches/Fuhrman 1/{self.imageName[:-4]}_{counter_1}.jpg', imgCrop_clear)
                        elif (self.imageProps[k]['region_attributes']['name'].title() == 'Fuhrman 2'):
                            counter_2 += 1
                            imgCrop = img[y:y+h, x:x+w]
                            cv2.imwrite(f'../dataset/Extracted Annotation Data/Image Patches/Fuhrman 2/{self.imageName[:-4]}_{counter_2}.jpg', imgCrop)
                            imgCrop_clear = imgCleared[y:y+h, x:x+w]
                            cv2.imwrite(f'../dataset/Extracted Annotation Data/Cleared Image Patches/Fuhrman 2/{self.imageName[:-4]}_{counter_2}.jpg', imgCrop_clear)
                        elif (self.imageProps[k]['region_attributes']['name'].title() == 'Fuhrman 3'):
                            counter_3 += 1
                            imgCrop = img[y:y+h, x:x+w]
                            cv2.imwrite(f'../dataset/Extracted Annotation Data/Image Patches/Fuhrman 3/{self.imageName[:-4]}_{counter_3}.jpg', imgCrop)
                            imgCrop_clear = imgCleared[y:y+h, x:x+w]
                            cv2.imwrite(f'../dataset/Extracted Annotation Data/Cleared Image Patches/Fuhrman 3/{self.imageName[:-4]}_{counter_3}.jpg', imgCrop_clear)
                
                # Step 3. Visualizing the annotated regions
                if display_annotations==True:
                    for cntr in contours:
                        x, y, w, h = cv2.boundingRect(cntr)
                        for k in range(len(self.imageProps)):
                            if (self.imageProps[k]['region_attributes']['name'].title() == 'Fuhrman 1'):
                                (w1, h1), _ = cv2.getTextSize('Fuhrman 1', cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
                            
                                cv2.rectangle(img, (x, y-20), (x+w1, y), (0, 255, 0), -1)
                                cv2.putText(img, 'Fuhrman 1', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                                cv2.rectangle(imgBW_filled, (x, y-20), (x+w1, y), (0, 255, 0), -1)
                                cv2.rectangle(imgBW_filled, (x, y), (x+w, y+h), (0, 255, 0), 1)
                                cv2.putText(imgBW_filled, 'Fuhrman 1', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                            
                            if (self.imageProps[k]['region_attributes']['name'].title() == 'Fuhrman 2'):
                                (w1, h1), _ = cv2.getTextSize('Fuhrman 2', cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
                                
                                cv2.rectangle(img, (x, y-20), (x+w1, y), (0, 255, 255), -1)
                                cv2.putText(img, 'Fuhrman 2', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 1)
                                cv2.rectangle(imgBW_filled, (x, y-20), (x+w1, y), (0, 255, 255), -1)
                                cv2.putText(imgBW_filled, 'Fuhrman 2', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                                cv2.rectangle(imgBW_filled, (x, y), (x+w, y+h), (0, 255, 255), 1)
                            
                            elif (self.imageProps[k]['region_attributes']['name'].title() == 'Fuhrman 3'):
                                (w1, h1), _ = cv2.getTextSize('Fuhrman 3', cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
                                
                                cv2.rectangle(img, (x, y-20), (x+w1, y), (0, 0, 255), -1)
                                cv2.putText(img, 'Fuhrman 3', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                                cv2.rectangle(imgBW_filled, (x, y-20), (x+w1, y), (0, 0, 255), -1)
                                cv2.putText(imgBW_filled, 'Fuhrman 3', (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                                cv2.rectangle(imgBW_filled, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    
                    plt.figure(figsize=(12,6))
                    plt.title(f'Image Name: {self.imageName}')
                    # Converting the BGR channel order to RGB for display
                    plt.imshow(img[:, :, ::-1])
                    plt.xticks([]), plt.yticks([])
                    plt.show()
                    
                    # Visualizing the extraction results
                    plt.figure(figsize=(12,6))
                    plt.title(f'{self.imageName} - Annotation Mask')
                    plt.imshow(imgBW_filled[:, :, ::-1])
                    plt.xticks([]), plt.yticks([])
                    plt.show()


class OptunaHandler:
    study_case_results_path=os.path.join('','optuna')
    study_results=dict()

    @staticmethod
    def flush():
        OptunaHandler.results.clear()

    @staticmethod
    def add_result(combination,metric_name,metric_value):
        if OptunaHandler.study_results.get(combination,None)==None:
            OptunaHandler[combination]=dict()
        OptunaHandler[combination][metric_name]=metric_value

    def __init__(self,new_study_case):
        self.study_id=new_study_case
        self.study = None
        self.trial = None
        self.study_objective_dimensions=list()

    def add_objective_dimension(self,param_name,direction):
        self.study_objective_dimensions[(param_name,direction)]
    
    def set_callback(self,callback_function):
        del self.study
        del self.trial
        if self.study_objective_dimensions==[]:
            raise AttributeError("No objective dimension has been setted properly")

        self.study=optuna.create_study(directions=[direction for _,direction in self.study_objective_dimensions],load_if_exists=True)
        self.study.optimize(callback_function,n_trials=50)
    
    def export(self):
        self.study.save_study_direction()
        self.study.trials_dataframe().to_csv(os.path.join(OptunaHandler.study_case_results_path,f'{self.study_id}.csv'))

    def export_best(self):
        if not os.path.exists(os.path.join(OptunaHandler.study_case_results_path,self.study_id)):
            os.mkdir(os.path.join(OptunaHandler.study_case_results_path,self.study_id))
        
        with open(os.path.join(OptunaHandler.study_case_results_path,self.study_id,'best_trial.pkl'), 'wb') as f:
            pickle.dump(self.study.best_trial, f)

        # save the best parameters
        with open(os.path.join(OptunaHandler.study_case_results_path,self.study_id,'best_params.pkl'), 'wb') as f:
            pickle.dump(self.study.best_params, f)

        # save the best value of the objective function
        with open(os.path.join(OptunaHandler.study_case_results_path,self.study_id,'best_value.pkl'), 'wb') as f:
            pickle.dump(self.study.best_value, f)