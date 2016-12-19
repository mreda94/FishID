from os import listdir
from os.path import isfile, join
import pylab as pl
import numpy as np
from skimage import io, filters, color, exposure
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

path="C:/Users/mreda_000/Desktop/train"
files=[]
labels = []
for i in ["ALB", "BET", "DOL", "LAG","NoF","OTHER","SHARK","YFT"]:
    files= files + [f for f in listdir("%s/%s" % (path, i)) if isfile(join("%s/%s" % (path, i), f))]
    labels= labels + [i for p in listdir("%s/%s" % (path, i)) if isfile(join("%s/%s" % (path, i), p))]
print(files)
print(labels)
def images(path):
    x=[]
    for i in range(len(files)):
        x=x+["%s/%s/%s" % (path, labels[i], files[i])]
        #print("%s/%sResized/%s.Bmp" % (path, name, i))
    x = io.ImageCollection(x)
    return x
x=[]
img=[]
for i in range(10):
    n = color.rgb2gray(images(path)[i])
    n= (n-np.mean(n))/np.std(n)
    m = filters.sobel(n)
    #fd = exposure.equalize_hist(m)
    img = img +[m]
    x = x + [np.ravel(m)]

io.imshow(img[4])



