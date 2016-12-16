# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:18:19 2016

@author: mreda_000
"""



import pylab as pl
import numpy as np
from skimage import io, filters, color
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
#path to the folder that holds my files
path = "C:/Users/mreda_000/Desktop"
brandonPath = "/Users/brandondalton/Documents/USC - Programming"
labelsInfo = read_csv("%s/trainLabels.csv" % path)
#extracts images from file
def images(name, path):
    x=[]
    for i in labelsInfo.ID:
        x=x+["%s/%sResized/%s.Bmp" % (brandonPath, name, i)]
        #print("%s/%sResized/%s.Bmp" % (path, name, i))
    x = io.ImageCollection(x)
    return x
x=[]
p=[]
#preprossed images through edge extraction, make array 1 dimentional and list of labels 
for i in range(len(images("train", brandonPath))):
    
    n=color.rgb2gray(images("train", brandonPath)[i])
    n=filters.sobel(n)
    x= x+[np.ravel(n)]
    p=p+[ord(labelsInfo.Class[i]) ]



# number of samples and height by width in pixel of photos
n_samples, h, w = len(x), 20, 20

np.random.seed(42)
#n_features is number of pixel in image
n_features = 400

# the label to predict is the id of the person
y = p
target_names = p
n_classes = len(target_names)

# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Compute a PCA unsupervised feature extraction / dimensionality reduction
n_components = 250

pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

eigenimages = pca.components_.reshape((n_components, h, w))

#Projecting the input data on the eigenimages orthonormal basis
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a SVM classification model

param_grid = {
         'C': [1, 1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced' ), param_grid)
clf = clf.fit(X_train_pca, y_train)
print(clf.best_estimator_)

#Train a KNN classification model using images before demensional reduction

param_grid1 = {
              'n_neighbors':[1, 3, 5, 7, 9, 11, 13]
            }
klf = GridSearchCV(KNeighborsClassifier(), param_grid1)

klf = klf.fit(X_train, y_train)
# Quantitative evaluation of the model quality on the test set

#prediction based on SVM
y_preds = clf.predict(X_test_pca)

#prediction based on KNN
y_predk = klf.predict(X_test)

#Get target names for y_test
target_name = []
for i in range(len(target_names)):
   target_name= target_name+[chr(target_names[i])]

print("Classification report for SVM:", classification_report(y_test, y_preds, target_names=target_name))
                            
print("The accuracy of SVM: %s" % clf.score(X_test_pca, y_test))

print("Classification report for KNN:", classification_report(y_test, y_predk, target_names=target_name))

print("The accuracy of KNN: %s" % klf.score(X_test, y_test))

# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images,titles, h, w, n_row=5, n_col=5):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = chr(y_pred[i])
    true_name = chr(y_test[i])
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_predk, y_test, target_names, i)
                         for i in range(len(y_preds))]

plot_gallery(X_test, prediction_titles, h, w)

#plot the gallery of the most significative eigenfaces

eigenimages_titles = ["eigenimages %d" % i for i in range(len(eigenimages))]
plot_gallery(eigenimages, eigenimages_titles, h, w)

pl.show()
