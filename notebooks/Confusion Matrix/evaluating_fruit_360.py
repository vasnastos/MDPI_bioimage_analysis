import os
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from data_preprocessing import validation_generator, batchSize


# ----------------------------------------- Step. 1 Loading the Trained Model ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

classifier = load_model("fruits_360_cnn.h5")

num_of_validationSamples = validation_generator.samples

imgWidth, imgHeight, colorDepth = (32, 32, 3)

# Gathering the class labels in a list
classesLabels = validation_generator.class_indices

classesLabels = {v: k for k, v in classesLabels.items()}
classes = list(classesLabels.values())


# ------------------------------------------- Step. 2 Classification Results -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Making predictions on the validation set. 'axis = 1' reads the predictions as 2D matrix
yPred = classifier.predict_generator(validation_generator, num_of_validationSamples // batchSize + 1)
yPred = np.argmax(yPred, axis=1)

# Printing the confusion matrix on the console
print("\nConfusion matrix:")
print(confusion_matrix(validation_generator.classes, yPred))

# Printing the classification report
print("\nClassification report:")
print(classification_report(validation_generator.classes, yPred, target_names=classes))


# ---------------------------------------- Step. 3 Visualizing Confusion Matrix ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class_accuracy = sum(yPred == validation_generator.classes) / len(validation_generator.classes)

plt.figure(figsize=(15, 15))
cnf_matrix = confusion_matrix(validation_generator.classes, yPred)
plt.imshow(cnf_matrix, interpolation='nearest')
plt.title("Classification Accuracy - {0:.1f}%".format(round(class_accuracy * 100)))
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)

plt.autoscale(enable=True, axis='both', tight=None)
# plt.tight_layout()
plt.show()


# ------------------------------------------ Step. 4 Visualizing Predictions ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def displayTesting(name, prediction, imput_image, true_label):
    blackBackground = [0, 0, 0]

    expandedImage = cv2.copyMakeBorder(imput_image, top=160, bottom=0, left=0, right=500,
                                       borderType=cv2.BORDER_CONSTANT, value=blackBackground)

    cv2.putText(expandedImage, "Predicted - " + str(prediction), org=(20, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

    cv2.putText(expandedImage, "True - " + str(true_label), org=(20, 120),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow(name, expandedImage)


def getRandomImage(path, imgWidth, imgHeight):
    # Loading random images from random folders from the validation parent folder
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))

    randomDirectory = np.random.randint(0, len(folders))
    pathClass = folders[randomDirectory]
    filePath = path + pathClass

    fileNames = [f for f in listdir(filePath) if isfile(join(filePath, f))]
    random_fileIndex = np.random.randint(0, len(fileNames))
    imageName = fileNames[random_fileIndex]

    finalPath = filePath + "/" + imageName

    return image.load_img(finalPath, target_size=(imgWidth, imgHeight)), finalPath, pathClass


# Making and gathering predictions in empty arrays
files = []
predictions = []
trueLabels = []

for i in range(0, 10):
    path = '../Course Resources/12. Optimizers, Adaptive Learning Rate & Callbacks/fruits-360/validation/'

    img, finalPath, true_label = getRandomImage(path, imgWidth, imgHeight)
    files.append(finalPath)
    trueLabels.append(true_label)

    x = image.img_to_array(img)
    x = x * 1./255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    indentClasses = classifier.predict_classes(images, batch_size=10)
    predictions.append(indentClasses)

for i in range(0, len(files)):
    image = cv2.imread((files[i]))
    displayTesting("Prediction", classesLabels[predictions[i][0]], image, trueLabels[i])
    cv2.waitKey(2 * 1000)

cv2.destroyAllWindows()