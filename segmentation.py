import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from skimage import color, exposure, feature, filters, measure, segmentation
from skimage.segmentation import clear_border
from skimage.measure import regionprops_table
from scipy import ndimage as ndi
from dataset import Dataset



# ROI: Region Of Interest
def segmenting_ROI(img_file, clear_patches=False):
    img_hsv = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)
    
    # Stage 1: Enhancing grayscale contrast by clipping pixel values to lower and upper STD
    mean, STD = cv2.meanStdDev(img_hsv[:,:,2])
    offset = 1
    img_v_enhanced = np.clip(img_hsv[:,:,2], mean - offset*STD, mean + offset*STD).astype(np.uint8)
    # Applying Gaussian blur to enhance edges
    kernel_size = (3,3)
    blur = cv2.GaussianBlur(img_v_enhanced, kernel_size, 0)
    
    # Stage 2. K-means clustering for binary cell extraction
    z = blur.reshape((-1, 1))
    z = np.float32(z)
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.2) 
    _, labels, centers = cv2.kmeans(data=z, K=K, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers) 
    res = centers[labels.flatten()]
    img_v_kmeans = res.reshape((blur.shape))
    thresh = clear_border(np.uint8(255 - (img_v_kmeans>np.min(centers))))
    
    # Stage 3: Floodfilling initial contour-segmented regions
    mask_filled = np.ones(thresh.shape, np.uint8)
    contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    for i, cnt in enumerate(contours):
        mask_filled_copy = mask_filled.copy()
        cv2.fillPoly(mask_filled_copy, pts=[cnt], color=(255,255,255))
        mask_filled = cv2.bitwise_or(mask_filled, mask_filled_copy)
        
    # Stage 4: Removing non-circular objects < 10 diameter value and smoothing edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    opening = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Stage 5: Applying watershed segmentation for dividing merged ROIs
    thresholds = filters.threshold_multiotsu(opening, classes=2)
    cells = opening > thresholds[0]

    distance = ndi.distance_transform_edt(cells)
    local_max_coords = feature.peak_local_max(distance, min_distance=7)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segm_cells = segmentation.watershed(-distance, markers, mask=cells)
    
    # Stage 6: Extracting final contour and bounding-box coordinates
    contour_coords = []
    bbox_coords = []
    cropped_image_samples = []
    
    num_objects = 0
    for c in np.unique(segm_cells):
        if c == 0:
            continue
        mask = np.zeros(opening.shape, dtype="uint8")
        mask[segm_cells == c] = 255
        prop = measure.regionprops(mask, opening)
        area = prop[0].area
        eccentricity = prop[0].eccentricity
        if 200 < area < 2000 and eccentricity < 0.95:            
            # Applying active countour models (ACMs) to the Watershed objects
            contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            # Determining the ACM segmention bounding boxes
            contour_coords.extend(contours)
            for cntr in contours:
                num_objects += 1
                x, y, w, h = cv2.boundingRect(cntr)
                bbox = (x, y), (x+w, y+h)
                bbox_coords.append(bbox)
                
                if clear_patches==True:
                    # A. Isolating cleared image patch for CNN classification
                    img_file_cleared = img_file.copy()
                    img_file_cleared[mask==0] = 0
                    cropped_image_samples.append(img_file_cleared[y:y+h, x:x+w])
                else:
                    # B. Isolating image patch for CNN classification
                    cropped_image_samples.append(img_file[y:y+h, x:x+w])
                    
    print(f'\nNumber of segmented objects: {str(num_objects)}\n')
    return img_v_kmeans, opening, segm_cells, contour_coords, bbox_coords, cropped_image_samples


def visualizing_preprocessing(image, title:str):
    img_kmeans, binary_opening, watershed_segm, contour_coords, bbox_coords, _, = segmenting_ROI(image)
    images = [image, img_kmeans, binary_opening, watershed_segm]
    labels = ['Original Color Image', 'HSV Value K-means (K=3)', 'Binary Regions of Interest (ROIs)', 'Watershed Semgented Regions']
    
    fig = plt.figure(figsize=(15,8))
    fig.suptitle(f"{title.replace('.jpg','')} Preprocessing Results", fontname='Times New Roman', fontsize=20)
    for i, _ in enumerate(labels):
        ax = plt.subplot(2, 2, i+1)
        ax.set_title(labels[i], fontname='Times New Roman')
        ax.axis('off')
        if len(images[i].shape) == 3:
            ax.imshow(images[i][:,:,::-1])
        elif 'Watershed' in labels[i]:
            ax.imshow(color.label2rgb(images[i], bg_label=0))
        else:
            ax.imshow(images[i], cmap='gray')
            
    plt.figure(figsize=(15,8))
    img_contours = image.copy()
    plt.suptitle(f"{title.replace('.jpg','')} - Active Contour Segmented Objects", fontname='Times New Roman', fontsize=20)
    cv2.drawContours(img_contours, contour_coords, -1, (0, 255, 0), 2)
    plt.imshow(img_contours[:,:,::-1])
    plt.show()
    
    plt.figure(figsize=(15,8))
    img_bboxes = image.copy()
    plt.suptitle(f"{title.replace('.jpg','')} - Bounding Box Detected Objects", fontname='Times New Roman', fontsize=20)
    for i in range(len(bbox_coords)):
        cv2.rectangle(img_bboxes, bbox_coords[i][0], bbox_coords[i][1], (0, 255, 0), 2)
    plt.imshow(img_bboxes[:,:,::-1])
    plt.show()
    

def cnn_detecting_objects(model,dataset, image_file, image_title:str, clear_patches):
    _, _, _, contours, bboxes, image_patches = segmenting_ROI(image_file, clear_patches)
    input_size = tuple(model.input.shape)[1:3]

    # Stage 1: Classify image patches
    # image_patches_tensor = np.array([tf.image.resize(images=tf.convert_to_tensor(cv2.bilateralFilter(image_patches[i],d=6,sigmaColor=75,sigmaSpace=75), dtype=tf.float32), size=input_size, method='lanczos5', preserve_aspect_ratio=False) for i in range(len(image_patches))])
    image_patches_tensor = np.array([tf.image.resize(images=tf.convert_to_tensor(image_patches[i]), size=input_size, method='lanczos5', preserve_aspect_ratio=False) for i in range(len(image_patches))])
    print(f'\nTensor shape: {image_patches_tensor.shape}, dtype={image_patches_tensor[0].dtype}')
    
    # TODO
    # if classification == "deep":
        # ... line 149 and beyond
    # elif classification == "conventional":
    #   a. call feature extractor for each 'image_patches tensor'
    #   b. apply all feature data preprocessing in the conventional model
    #   c. classify the samples (retrieve label and class prop)

    tprobs = np.array(model.predict(image_patches_tensor, batch_size=16))
    tpreds = tprobs.argmax(axis=1)
    
    # Stage 2: Gathering results
    column_names = ['ContourCoords', 'BBoxCoords', 'PredClass', 'PredProb']
    predicted_samples = pd.DataFrame(index=range(image_patches_tensor.shape[0]), columns=range(len(column_names)))
    predicted_samples.columns = column_names
    
    for i in range(len(predicted_samples)):
        predicted_samples['ContourCoords'][i] = contours[i]
        predicted_samples['BBoxCoords'][i] = bboxes[i]
        predicted_samples['PredClass'][i] = dataset.class_names[tpreds[i]]
        predicted_samples['PredProb'][i] = float('{:.2f}'.format(tprobs[i][tpreds][i]))
        
    # Stage 3: Retrieve class label ratios and most-dominant class
    fuhrman_ratios = dict()
    for i in range(dataset.num_classes):
        ratio = len(predicted_samples[predicted_samples['PredClass'] == f'Fuhrman {i+1}']) / len(predicted_samples)
        fuhrman_ratios[f'Fuhrman {i+1}'] = float('{:.2f}'.format(ratio * 100))
    
    max_value, max_label = max(zip(fuhrman_ratios.values(), fuhrman_ratios.keys()))
    plt.figure(figsize=(8,8))
    plt.suptitle(f'RCC Stage: {max_label}, {max_value}%', fontname='Times New Roman', fontsize=20)
    pps = plt.bar(range(len(fuhrman_ratios)), list(fuhrman_ratios.values()), color=['green', 'yellow', 'red'], align='center')
    for p in pps:
        height = p.get_height()
        plt.annotate( "{}%".format(height), (p.get_x() + p.get_width()/2, height+.05), ha="center", va="bottom", fontname='Times New Roman', fontsize=15)
    plt.xticks(range(len(fuhrman_ratios)), list(fuhrman_ratios.keys()))
    plt.show()
    
    # Stage 4: Visualizing RCC staging results
    img_contours = image_file.copy()
    img_bboxes = image_file.copy()
    
    plt.figure(figsize=(12,8))
    plt.suptitle(f"{image_title.replace('.jpg','')} - Segmented Nuclei - RCC Stage: {max_label}, {max_value}%", fontname='Times New Roman', fontsize=20)
    for i in range(len(predicted_samples)):
        if predicted_samples['PredClass'][i] == 'Fuhrman 1':
            cv2.drawContours(img_contours, predicted_samples['ContourCoords'][i], -1, (0, 255, 0), 2)
        elif predicted_samples['PredClass'][i] == 'Fuhrman 2':
            cv2.drawContours(img_contours, predicted_samples['ContourCoords'][i], -1, (0, 255, 255), 2)
        elif predicted_samples['PredClass'][i] == 'Fuhrman 3':
            cv2.drawContours(img_contours, predicted_samples['ContourCoords'][i], -1, (0, 0, 255), 2)
    plt.imshow(img_contours[:,:,::-1])
    
    plt.figure(figsize=(12,8))
    plt.suptitle(f"{image_title.replace('.jpg','')} - Detected Nuclei - RCC Stage: {max_label}, {max_value}%", fontname='Times New Roman', fontsize=20)
    for i in range(len(predicted_samples)):
        x1, y1, w1, h1 = predicted_samples['BBoxCoords'][i][0][0], predicted_samples['BBoxCoords'][i][0][1], predicted_samples['BBoxCoords'][i][1][0], predicted_samples['BBoxCoords'][i][1][1]
        (w2, h2), _ = cv2.getTextSize(f"F_, {predicted_samples['PredProb'][i]}", cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
        if predicted_samples['PredClass'][i] == 'Fuhrman 1':
            cv2.rectangle(img_bboxes, (x1, y1-(h2+2)), (x1+w2, y1), (0, 255, 0), -1)
            cv2.putText(img_bboxes, f"F1, {predicted_samples['PredProb'][i]}", (x1, y1-2), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
            cv2.rectangle(img_bboxes, (x1, y1), (w1, h1), (0, 255, 0), 1)
        elif predicted_samples['PredClass'][i] == 'Fuhrman 2':
            cv2.rectangle(img_bboxes, (x1, y1-(h2+2)), (x1+w2, y1), (0, 255, 255), -1)
            cv2.putText(img_bboxes, f"F2, {predicted_samples['PredProb'][i]}", (x1, y1-2), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
            cv2.rectangle(img_bboxes, (x1, y1), (w1, h1), (0, 255, 255), 1)
        elif predicted_samples['PredClass'][i] == 'Fuhrman 3':
            cv2.rectangle(img_bboxes, (x1, y1-(h2+2)), (x1+w2, y1), (0, 0, 255), -1)
            cv2.putText(img_bboxes, f"F3, {predicted_samples['PredProb'][i]}", (x1, y1-2), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
            cv2.rectangle(img_bboxes, (x1, y1), (w1, h1), (0, 0, 255), 1)
    plt.imshow(img_bboxes[:,:,::-1])
    plt.show()


def staging_all_images(self):
    pass


if __name__ == '__main__':
    # img_path = os.path.join('..','..','Dataset','40x','Fuhrman 1','95-19-B15-2.jpg')
    # img_path = os.path.join('..','..','Dataset','40x','Fuhrman 2','53-19-D-1.jpg')
    img_path = os.path.join('..','..','Dataset','40x','Fuhrman 3','2367-19-H-1.jpg')
    
    visualizing_preprocessing(cv2.imread(img_path), title=os.path.basename(img_path))

