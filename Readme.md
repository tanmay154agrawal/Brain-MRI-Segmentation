
# Brain tumour classification and segmentation

Mri Tumour Segmentation is an ongoing area of research
in the field of medical image analysis . Accurate and early
detection of tumour is essential for treatment planning and
prognosis.In this project firstly simple Machine learning
model is applied to classify the image into having tumour or
not and comparison is made on accuracy of different handcrafted feature. Then, different image segmentation techniques are applied to seperate out the tumour region from
the brain.





## Deployment

this code is performing image processing operations to enhance the quality of MRI images, including resizing, thresholding, morphological operations, and bitwise operations. The stripped images are then stored in a set along with their corresponding indices. The original images are also stored in a separate list.

```bash

mri_image=[]
set=[]
for mri,i in images:
   img=cv2.resize(mri, (256,256), interpolation=cv2.INTER_LINEAR)
   _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
   kernel = np.ones((4,4), np.uint8)
   closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)
   kernel = np.ones((3,3), np.uint8)  
   eroded = cv2.erode(closing, kernel, iterations=10)
   stripped = cv2.bitwise_and(img, eroded)
   cv2_imshow(stripped)
   set.append((stripped,i))
   mri_image.append(mri)

```

the below code is a data augmentation step to create a balanced dataset to avoid bias.we adjust the image's contrast using the cv2.convertScaleAbs() function. This function linearly scales pixel values in the input image by multiplying them with a scaling factor (alpha) and then adding a constant (beta). In this case, the scaling factor is set to 1.5 and the constant is set to 0. This will make the dark pixels in the image darker and the bright pixels brighter, enhancing the contrast of the image.


```bash

for idx,mri in enumerate(set):
  mri,i= mri
  if i==0 and idx%3==0:
    img_adjusted = cv2.convertScaleAbs(mri, alpha=1.5, beta=0)
    set.append((img_adjusted,i))

```


The below code uses K means segmentation where the  algorithm groups similar pixels in the image into K clusters based on their pixel intensity values in indentify our region of interest. 


```bash

segment=[]
for mri,i in set:
  if i==1:
    mri1=mri.copy()
    data = np.float32(mri1.reshape((-1, 1)))
    K = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    intensity = np.zeros(K)
    for i in range(K):
       cluster_pixels = data[label == i]
       intensity[i] = np.mean(cluster_pixels)
    idx=np.argmax(intensity)
    label=label.reshape(mri.shape)
    mask = np.uint8(label==idx)
    mri[mask==0] = [0]
    cv2_imshow(mri1)
    segment.append(mri1)


```


This code is finding the largest contour within the segmented image and storing it in a binary mask image. Additionally, the code is computing the size of the largest contour and storing it in a list called tumour_size.


```bash

tumours=[]
tumour_size=[]
for tumour in segment: 
  contours, hierarchy = cv2.findContours(tumour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = None
  max_area = 0
  img=mask = np.zeros_like(tumour)
  for contour in contours:
      area = cv2.contourArea(contour)
      if area > max_area:
          largest_contour = contour
          max_area = area
  tumour_size.append(max_area)
  cv2.fillPoly(img, pts=[largest_contour], color=255)
  tumours.append(img)


```


This code is segmenting tumor regions in MRI images using the watershed algorithm. The algorithm identifies local maxima in the distance transform of a binary mask and uses them as markers to partition the image into regions. T



```bash

for mri,i in set:
  if i==1:
    thresh_value = 120
    binary = mri > thresh_value
    distance = ndimage.distance_transform_edt(binary)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=binary) 
    markers = ndimage.label(local_maxi)[0] 
    labels = watershed(distance, markers, mask=binary)

```

The below code applies several image processing techniques  to enhance the image and extract features that can be used for classification. These techniques include morphological opening and closing, distance transformation, and connected component analysis to identify markers that are used to segment the image. The resulting markers are flattened into a 1D array that serves as the feature vector for the image.based on this features we classify the images 

```bash

for mri,i in set:
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
  opening = cv2.morphologyEx(mri, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
  dist_transform = cv2.distanceTransform(mri, cv2.DIST_L2, 5)
  ret, markers = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, cv2.THRESH_BINARY)
  markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
  features = np.ravel(markers)
  X.append(features)
  Y.append(i)

```

The below code is computes three image features (entropy, standard deviation, and skewness) for each MRI image in the set. with the help of these features we classify the images.



```bash

for mri,i in set:
  ent=entropy(mri)
  ent = np.where(np.isnan(ent), 0, ent)
  std_dev = np.std(mri)
  std_dev = np.where(np.isnan(std_dev), 0, std_dev)
  sk = skew(np.ravel(mri))
  sk = np.where(np.isnan(sk), 0, sk)
  X.append((ent,std_dev,sk))
  Y.append(i)


```
## dataset details

We used Brain MRI Images for Brain Tumor Detection
from Kaggle. It has 253 images in which 155 images have
tumours and 98 images doesn’t . We used data augmentation to take care of class imbalance.
