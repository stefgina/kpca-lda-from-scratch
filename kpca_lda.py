import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import decomposition
from sklearn import model_selection
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import  KernelPCA
from sklearn.utils import shuffle
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.spatial.distance import pdist, squareform
import sys
from scipy.linalg import eigh
import numpy as np
import scipy.spatial.distance
import scipy.linalg
import sklearn.metrics.pairwise
from scipy import exp
import sklearn.preprocessing
import struct
from sklearn.metrics.pairwise import euclidean_distances
np.set_printoptions(threshold=sys.maxsize)

####################################################
####################################################
##########[ CUSTOM FUNCTIONS ]######################
####################################################
####################################################


# the RBF kernel I made
# its getting used on kpca() and project_kpca()
def rbfkernel(gamma, distance):
    return np.exp(-gamma * distance)

# KPCA function I made for FITTING/TRANSFORMING the TRAIN
def kpca(X, gamma=1, components=300):
    
 
    # compute square dist (point to point)
    sq_dist=scipy.spatial.distance.pdist(X,'sqeuclidean')
    
    # reshape sqare dist
    reshape=scipy.spatial.distance.squareform(sq_dist)
    
    # using the function i made with the rbf kernel
    K_old=rbfkernel(gamma,reshape)
    
    # centering the kernel using sklearn
    kern_cent=sklearn.preprocessing.KernelCenterer()
    K=kern_cent.fit_transform(K_old)

    # eigenvecs an eigenvalues of KERNEL MATRIX (ONLY THE TOP N_COMP)
    eigenval, eigenvec = scipy.linalg.eigh(K,eigvals=(K.shape[0]-components, K.shape[0]-1))
    
    # sorting from biggest to lowest
    idx = np.argsort((eigenval))
    eigenvec=eigenvec[:,idx]
    eigenval=eigenval[idx]
    
    # arranging into arrays and vectors
    W=[]
    L=[]
    for i in range(1,components+1):
        W.append(eigenvec[:,-i])
        L.append(eigenval[-i])

    alphas=np.column_stack(W)
    lambdas=np.array(L)
    X_transformed=alphas*np.sqrt(lambdas)
    
    # a) returning the train projections
    # b) returning the top eigenvectors, we will use them for
    # test projections.
    return (X_transformed,alphas)



# porjection KPCA function to transform the test data                          
def project_kpca(Y,X, gamma, D):
    
    
    # distance of TEST and TRAIN matrices
    dist = euclidean_distances(Y,X)
    
    # elevating distances into high RBF KERNEL
    k = rbfkernel(gamma, dist)
    
    #calculating projections of k on the top Eigenvectors.
    projections = np.dot(k, D)
    
    
    return np.array(projections)


####################################################
####################################################
##########[ MAIN STARTS HERE ]######################
####################################################
####################################################


### IMAGES (TRAIN & TEST) ###

# First 16 bytes sto kathe idx3 einai stoixeia megethous kai typou dedomenwn, ola ta ypoloipa einai pixels.
# Kathe unsigned int <I> pianei 4b opote kanw ta prwta 16b unpack, 
# wste na parw ta aparaithta stoixeia (diastaseis tou pinaka pou tha ftiaxw) kai na mou meinoun mono ta pixel bytes sto array.
# Gnwsto apo to site, pws xrhsimopoieitai big-endian seira apothikeushs eksou kai to ">" sto unpacking kai to byteorder.

with open('mnist/train-images.idx3-ubyte','rb') as f: # rb - read binary
    magicTrain, sizeTrain, nrows, ncols = struct.unpack(">IIII", f.read(16)) 
    trainImages = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>')) # ftiaxnw ena array kvantizontas ana 8 bit.
    trainImages = trainImages.reshape((sizeTrain, nrows*ncols)) # reshape stis 60,000 x 256 wste na einai diaxeirismo.
    
with open('mnist/t10k-images.idx3-ubyte','rb') as f:
    magicTest, sizeTest, nrows, ncols = struct.unpack(">IIII", f.read(16)) 
    testImages = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>')) # same here..
    testImages = testImages.reshape((sizeTest, nrows*ncols))

### LABELS (TRAIN & TEST) ###

# First 8 bytes auth th fora einai stoixeia megethous kai typou dedomenwn, ola ta ypoloipa einai labels.
# Authn thn fora den xreiazetai na apothikeusw kapou to unpack mou, kathws exw tis diastaseis apo prin
# opote apla to kanw gia na aferaisw ta prwta 8 bytes apo to arxeio.

with open('mnist/t10k-labels.idx1-ubyte','rb') as f:
    struct.unpack(">II", f.read(8))
    testLabels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>')) # no need for reshape einai hdh sthn morfh pou thelw.
    
with open('mnist/train-labels.idx1-ubyte','rb') as f:
    struct.unpack(">II", f.read(8))
    trainLabels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>')) # same here..

# In order for KPCA to work in memory bounds
# I keep the first 15k for training and 10k test  
trainImages, trainLabels = np.float32(trainImages[:15000])/255., np.float32(trainLabels[:15000])
testImages, testLabels = np.float32(testImages[:10000])/255., np.float32(testLabels[:10000])

trainImages = np.float32(trainImages)/255.
testImages = np.float32(testImages)/255.

# shuffle the dataset in UNISON mode (together)
trainImages, trainLabels = shuffle(trainImages, trainLabels, random_state=1)
testImages, testLabels = shuffle(testImages, testLabels, random_state=1)

#trainImages = (trainImages - trainImages.mean())/trainImages.std()
#testImages = (testImages -testImages.mean())/testImages.std()

#trainImages = StandardScaler().fit_transform(train)
#testImages = StandardScaler().fit_transform(test)

print("The shape of train images is ", trainImages.shape )
print("The shape of test images is ", testImages.shape )


####### custom KPCA #######
###########################

# start the kpca timer
startKpca = int(round(time.time() * 1000))

print(" ")
print(" ---- STARTING my custom KPCA ---- ")

# MY kernel PCA (300 comp, gamma=1, RBF)
# outputs the projected TRAIN data on principal components, AND the top N eigenvectors
X_kpca, eigenvectors = kpca(trainImages)

print("The shape of train after xkpca ", X_kpca.shape )

# PROJECTING the TEST on the principal component space made by TRAIN
# in order NOT to cheat (etc. kpca on both)
X_test_pca = project_kpca(testImages,trainImages,1,eigenvectors)

print("The shape of test after xkpca ", X_test_pca.shape )

# stop the kpca timer
endKpca = int(round(time.time() * 1000))

print("---- My custom KPCA took ", (endKpca-startKpca), " ms----")


######## sklearn LDA #######
############################

# start the lda timer
startLda = int(round(time.time() * 1000))

print(" ")
print(" ---- STARTING sklearn's LDA ---- ")

# Using the sklearn LDA, drops to 9 components from 300 (classes - 1)
lda = LinearDiscriminantAnalysis()

# inputs from KPCA outputs
X_lda = lda.fit_transform(X_kpca,trainLabels)
print("The shape of train after kpca + lda ", X_lda.shape )

# only transform for test NO fit_transform, else is cheating.
X_test_lda = lda.transform(X_test_pca)
print("The shape of test after kpca + lda ", X_test_lda.shape )

# stop the lda timer
endLda = int(round(time.time() * 1000))

print("---- The sklearn LDA took ", (endLda-startLda), " ms----")


####### kNN classification with KPCA+LDA ############
#####################################################

startKnn = int(round(time.time() * 1000))
print(" ")
print(" Starting the Knn classification WITH")
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_lda, trainLabels)



endKnn = int(round(time.time() * 1000))
print("--KNN fitting finished in ", (endKnn-startKnn), "ms--")


expected = testLabels
predicted = clf.predict(X_test_lda)


print("RESULTS for kNN classifier %s:\n%s\n"
     % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


###### Nearest Centroid classification WITH KPCA LDA ########
##############################################################

# starting the centroid timer
startCent = int(round(time.time() * 1000))
print(" ")
print(" ---Starting Nearest Centroid WITH KPCA+LDA--- ")

centroidReducted = NearestCentroid()
centroidReducted.fit(X_lda, trainLabels)
NearestCentroid(metric='euclidean', shrink_threshold=None)

#stopping the centroid timer
endCent = int(round(time.time() * 1000))
print("----- Centroid fitting finished in ", (endCent-startCent), "ms---")

expected = testLabels
predicted = centroidReducted.predict(X_test_lda)


print("Results for Centroid classifier %s:\n%s\n"
     % (centroidReducted, metrics.classification_report(expected, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
