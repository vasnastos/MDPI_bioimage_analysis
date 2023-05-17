import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

from keras.models import load_model
from loading_simpsons_dataset import trainGenerator, validationGenerator, batchSize
from loading_simpsons_dataset import num_of_train_samples, num_of_validation_samples, num_of_classes


# ------------------------------------------- Step 1. Classification Report --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Loading the trained LittleVGG model
model = load_model("simpsons_little_vgg_15_epochs.h5")
print("\nPre-trained model loaded.")

classLabels = validationGenerator.class_indices
classLabels = {v: k for k, v in classLabels.items()}

classes = list(classLabels.values())

# Making predictions on the validation data
yPred = model.predict_generator(validationGenerator, num_of_validation_samples // batchSize + 1)
yPred = np.argmax(yPred, axis=1)

# Confusion matrix and classification report
confMatrix = confusion_matrix(validationGenerator.classes, yPred)
print("\nConfusion matrix: \n{}".format(confMatrix))

print("\nClassification report:")
print(classification_report(y_true=validationGenerator.classes, y_pred=yPred, target_names=classes))


# ---------------------------------------- Step 2. Visualizing Confusion Matrix ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Plotting the seaborn confusion matrix
cm_dataframe = pd.DataFrame(confMatrix, index=classes, columns=classes)

plt.figure()
sns.heatmap(cm_dataframe, annot=True, fmt="d", linewidths=5, square=True, cmap='Blues_r')
plt.title("LittleVGG - The Simpsons (conf. matrix)")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.autoscale(enable=True, axis='both', tight=None)
plt.show()


# Plotting the confusion matrix in colormap
plt.figure(figsize=(8, 8))
plt.imshow(confMatrix, interpolation='nearest')
plt.title("LittleVGG - The Simpsons (conf. matrix)")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes)
plt.autoscale(enable=True, axis='both', tight=None)
plt.tight_layout()
plt.show()