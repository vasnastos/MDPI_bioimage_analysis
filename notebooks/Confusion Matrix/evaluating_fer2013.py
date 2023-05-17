import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

import os
from os import listdir
from os.path import isfile, join

from loading_fer2013_dataset import validationGenerator, imgRows, imgCols
from loading_fer2013_dataset import batchSize, numOfClasses


# ----------------------------------------- Step 1. Loading Pre-Trained Model ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Loading the pre-trained LittleVGG classifier
model = load_model("fer2013_little_vgg_15_epochs.h5")


# Retrieving all class labels from the validation data
classLabels = validationGenerator.class_indices
classLabels = {v: k for k, v in classLabels.items()}
print("\nClass labels:", classLabels)

# Re-arranging the class labels into a list
classes = list(classLabels.values())


# --------------------------------------------- Step 2. Making Predictions ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

num_of_validationSamples = validationGenerator.n

# Running predictions on the validation dataset
yPred = model.predict_generator(validationGenerator, num_of_validationSamples // batchSize + 1)
yPred = np.argmax(yPred, axis=1)

# Printing the confusion matrix
confMatrix = confusion_matrix(validationGenerator.classes, yPred)
print("\nConfusion matrix: \n{}".format(confMatrix))

# Printing the classification report
print("\nClassification report: {}".format(classification_report(validationGenerator.classes, yPred,
                                                                 target_names=classes)))

# Plotting the seaborn confusion matrix
cm_dataframe = pd.DataFrame(confMatrix, index=classes, columns=classes)

plt.figure()
sns.heatmap(cm_dataframe, annot=True, fmt="d", linewidths=5, square=True, cmap='Blues_r')
plt.title("LittleVGG - Emotions")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.autoscale(enable=True, axis='both', tight=None)
plt.show()


# Plotting the confusion matrix in colormap
plt.figure(figsize=(8, 8))
plt.imshow(confMatrix, interpolation='nearest')
plt.title("LittleVGG - Emotion")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
plt.autoscale(enable=True, axis='both', tight=None)
plt.tight_layout()
plt.show()


# ------------------------------------------- Step 3. Displaying Predictions -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def displayTest(name, pred, img, true_label):
    blackBackground = [0, 0, 0]

    expandedImage = cv2.copyMakeBorder(img, 160, 0, 0, 300, cv2.BORDER_CONSTANT, value=blackBackground)
    cv2.putText(expandedImage, "True - " + true_label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(expandedImage, "Predited - " + pred, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(name, expandedImage)


# Defining a function that loads a random images from a random folder in the validation images path
def getRandomImage(path, img_width, img_height):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))

    path_class = folders[random_directory]
    file_path = path + path_class

    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path + "/" + image_name
    return image.load_img(final_path, target_size=(img_width, img_height), grayscale=True), final_path, path_class


files = []
predictions = []
trueLabels = []


# Running predictions on the random validation images
for i in range(0, 10):
    path = "../Course Resources/" \
           "18. Deep Surveillance - Build a Face Detector with Emotion, Age and Gender Recognition/" \
           "fer2013/validation/"

    randImage, finalPath, true_label = getRandomImage(path, imgRows, imgCols)
    files.append(finalPath)
    trueLabels.append(true_label)

    # Loading each image to a numpy array
    x = image.img_to_array(randImage)
    # Normalizing the image values (since the LittleVGG model is trained with normalized samples)
    x = x / 255.
    # Adding an extra dimension to form a Keras image tensor
    x = np.expand_dims(x, axis=0)

    # Providing more general stacking and concatenation operations
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    predictions.append(classes)

for i in range(0, len(files)):
    image = cv2.imread(files[i])
    image = cv2.resize(src=image, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    displayTest("Prediction", classLabels[predictions[i][0]], image, trueLabels[i])
    cv2.moveWindow("Prediction", 250, 250)
    cv2.waitKey(2 * 1000)

cv2.destroyAllWindows()