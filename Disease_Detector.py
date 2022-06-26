import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from skimage import data, color, feature
from skimage.feature import hog
import glob

def loadimage(arr,name_of_disease):
    label=[]
    strr = "rgb/"+"Tomato___"+name_of_disease+"/*.JPG"
    for file in glob.glob(strr):
        img=np.asarray(plt.imread(file))
        arr.append(img)
        label.append(name_of_disease)
    return arr,label

healthy=[]
Bacterial_spot=[]
Early_blight=[]
Late_blight=[]
Leaf_Mold=[]
Septoria_leaf_spot=[]
Spider_mites_Two_spotted_spider_mite=[]
Target_Spot=[]
Tomato_mosaic_virus=[]
Tomato_Yellow_Leaf_Curl_Virus=[]

healthy,label_healthy=loadimage(healthy,"healthy")
Bacterial_spot,label_Bacterial_spot=loadimage(Bacterial_spot,"Bacterial_spot")
Early_blight,label_Early_blight=loadimage(Early_blight,"Early_blight")
Late_blight,label_Late_blight=loadimage(Late_blight,"Late_blight")
Leaf_Mold,label_Leaf_Mold=loadimage(Leaf_Mold,"Leaf_Mold")
Septoria_leaf_spot,label_Septoria_leaf_spot=loadimage(Septoria_leaf_spot,"Septoria_leaf_spot")
Spider_mites_Two_spotted_spider_mite,label_Spider_mites_Two_spotted_spider_mite=loadimage(Spider_mites_Two_spotted_spider_mite,"Spider_mites_Two_spotted_spider_mite")
Target_Spot,label_Target_Spot=loadimage(Target_Spot,"Target_Spot")
Tomato_mosaic_virus,label_Tomato_mosaic_virus=loadimage(Tomato_mosaic_virus,"Tomato_mosaic_virus")
Tomato_Yellow_Leaf_Curl_Virus,label_Tomato_Yellow_Leaf_Curl_Virus=loadimage(Tomato_Yellow_Leaf_Curl_Virus,"Tomato_Yellow_Leaf_Curl_Virus")

X_Shapedes=np.concatenate((healthy,Bacterial_spot,Early_blight,Late_blight,Leaf_Mold,Septoria_leaf_spot,Spider_mites_Two_spotted_spider_mite,Target_Spot,Tomato_mosaic_virus,Tomato_Yellow_Leaf_Curl_Virus))
y_Shapedes=np.concatenate((label_healthy,label_Bacterial_spot,label_Early_blight,label_Late_blight,label_Leaf_Mold,label_Septoria_leaf_spot,label_Spider_mites_Two_spotted_spider_mite,label_Target_Spot,label_Tomato_mosaic_virus,label_Tomato_Yellow_Leaf_Curl_Virus))

X_train, X_test, y_train, y_test = train_test_split(X_Shapedes, y_Shapedes, test_size=0.2, random_state=154)

def preprocessing1(arr):
    arr_prep=[]
    for i in range(np.shape(arr)[0]):
        img=cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        img=resize(img, (72, 72),anti_aliasing=True)
        arr_prep.append(img)
    return arr_prep

def FtrExtractHOG(img):
    ftr,_=hog(img, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, multichannel=False)
    return ftr

def featureExtraction(arr):
    arr_feature=[]
    for i in range(np.shape(arr)[0]):
        arr_feature.append(FtrExtractHOG(arr[i]))
    return arr_feature

X_trainp=preprocessing1(X_train)
X_testp=preprocessing1(X_test)
X_trainftr=featureExtraction(X_trainp) 
X_testftr=featureExtraction(X_testp)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=11)
knn_clf.fit(X_trainftr, y_train)

y_knn_pred = knn_clf.predict(X_testftr)

print(accuracy_score(y_test, y_knn_pred)*100,'%')