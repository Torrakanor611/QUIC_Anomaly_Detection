import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn import svm
import sys, warnings
warnings.filterwarnings('ignore')


## GLOBAL VARIABLES
Classes={0:'Browsing', 1:'DoS'}
oClass = None
nfig = None

def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            

def plotFeatures(title,features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])
    
    plt.title(title)
    plt.xlabel("Mean (1st column)")
    plt.ylabel("Median (2nd column)")
    plt.show()
    waitforEnter(True)
    

def myPCA(trainFeaturesN,testFeaturesN, trainClass):
    pca = PCA(n_components=3, svd_solver='full')

    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)
    testFeaturesNPCA = trainPCA.transform(testFeaturesN)

    print(trainFeaturesNPCA.shape,trainClass.shape)
    plt.figure(nfig)
    plotFeatures("PCA",trainFeaturesNPCA,trainClass,0,1)

    return trainFeaturesNPCA,testFeaturesNPCA


def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))


def CentroidsDistance(trainClass,trainFeaturesN,testFeaturesN): 
    centroids={}
    pClass=(trainClass==0).flatten()
    centroids.update({0:np.mean(trainFeaturesN[pClass,:],axis=0)})
    print('All Features Centroids:\n',centroids)


    AnomalyThreshold=1.2
    print('\n-- Anomaly Detection based on Centroids Distances --')
    nObsTest,nFea=testFeaturesN.shape
    for i in range(nObsTest):
        x=testFeaturesN[i]
        dist=distance(x,centroids[0])
        if dist >AnomalyThreshold:
            result="Anomaly"
        else:
            result="OK"

        print('Obs: {:2} ({}): Normalized Distance to Centroids: {:.4f}   -> Result -> {}'.format(i,Classes[oClass[i][0]],dist,result))


def CentroidsDistance_PCA(trainClass,trainFeaturesNPCA,testFeaturesNPCA):
    centroids={}
    pClass=(trainClass==0).flatten()
    centroids.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
    print('All Features Centroids:\n',centroids)

    AnomalyThreshold=1.2
    print('\n-- Anomaly Detection based on Centroids Distances (PCA Features) --')
    nObsTest,nFea=testFeaturesNPCA.shape
    for i in range(nObsTest):
        x=testFeaturesNPCA[i]
        dist=distance(x,centroids[0])
        if dist > AnomalyThreshold:
            result="Anomaly"
        else:
            result="OK"
        
        print('Obs: {:2} ({}): Normalized Distance to Centroids (PCA): {:.4f} -> Result -> {}'.format(i,Classes[oClass[i][0]],dist,result))


def MultivariatePDF_PCA(trainClass, trainFeaturesNPCA,testFeaturesNPCA):
    print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
    means={}
    pClass=(trainClass==0).flatten()
    means.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
    print(means)
    print("\n-------------------------- COVS -------------\n\n")
    covs={}
    covs.update({0:np.cov(trainFeaturesNPCA[pClass,:],rowvar=0)})
    print(covs)

    AnomalyThreshold=0.05
    nObsTest,nFea=testFeaturesNPCA.shape
    for i in range(nObsTest):
        x=testFeaturesNPCA[i,:]
        prob = multivariate_normal.pdf(x,means[0],covs[0])
        if prob < AnomalyThreshold:
            result="Anomaly"
        else:
            result="OK"
        
        print('Obs: {:2} ({}): Probabilities: {:.4e} -> Result -> {}'.format(i,Classes[oClass[i][0]],prob,result))


def OneClassSVM_PCA(trainFeaturesNPCA, testFeaturesNPCA):
    print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
    ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesNPCA)  
    rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesNPCA)  
    poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesNPCA)  

    L1=ocsvm.predict(testFeaturesNPCA)
    L2=rbf_ocsvm.predict(testFeaturesNPCA)
    L3=poly_ocsvm.predict(testFeaturesNPCA)

    AnomResults={-1:"Anomaly",1:"OK"}

    nObsTest,nFea=testFeaturesNPCA.shape
    for i in range(nObsTest):
        print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[oClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


def OneClassSVM(trainFeaturesN, testFeaturesN):
    print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
    ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesN)  
    rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesN)  
    poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesN)  

    L1=ocsvm.predict(testFeaturesN)
    L2=rbf_ocsvm.predict(testFeaturesN)
    L3=poly_ocsvm.predict(testFeaturesN)

    AnomResults={-1:"Anomaly",1:"OK"}

    nObsTest,nFea=testFeaturesN.shape
    for i in range(nObsTest):
        print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[oClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

def main():
    plt.ion()
    ## FEATURE EXTRACTION
    ## -- Load features from browsing and attack -- ##
    features_browsing=np.loadtxt("Browsing_obs_features.dat")
    oClass_browsing=np.ones((len(features_browsing),1))*0

    features_dos=np.loadtxt("DoS_obs_features.dat")
    oClass_dos=np.ones((len(features_dos),1))*0

    features=np.vstack((features_browsing,features_dos))
    global oClass 
    oClass = np.vstack((oClass_browsing,oClass_dos))
    print('Train Stats Features Size:',features.shape)

    global nfig
    nfig = 1
    plt.figure(nfig)
    plotFeatures("Features",features,oClass,0,1)
    nfig += 1

    percentage=0.5

    ## FEATURE DATASETS FOR TRAINING AND TEST

    pB=int(len(features_browsing)*percentage)   ## Train with licit traffic (Browsing)
    trainFeatures_browsing=features_browsing[:pB,:]
    trainFeatures=np.vstack((trainFeatures_browsing))
    trainClass=np.vstack((oClass_browsing[:pB],))

    pD=int(len(features_dos)*percentage)        ## Test with mixed traffic (Browsing+DoS)
    testFeatures_dos=features_browsing[pD:,:]
    testFeatures_browsing=features_browsing[pB:,:]
    testFeatures=np.vstack((testFeatures_browsing,testFeatures_dos))
    testClass=np.vstack((oClass_browsing[pB:],oClass_dos[pD:]))


    ## FEATURE NORMALIZATION 
    trainScaler = MaxAbsScaler().fit(trainFeatures)
    trainFeaturesN=trainScaler.transform(trainFeatures)
    testFeaturesN=trainScaler.transform(testFeatures)


    ## FEATURE REDUCTION
    trainFeaturesNPCA,testFeaturesNPCA = myPCA(trainFeaturesN, testFeaturesN, trainClass)


    ## ANOMALY DETECTION Statistics
    CentroidsDistance(trainClass, trainFeaturesN, testFeaturesN)
    CentroidsDistance_PCA(trainClass, trainFeaturesNPCA, testFeaturesNPCA)
    MultivariatePDF_PCA(trainClass, trainFeaturesNPCA, testFeaturesNPCA)


    ## ANOMALY DETECTION Machine Learning
    OneClassSVM(trainFeaturesN, testFeaturesN)
    OneClassSVM_PCA(trainFeaturesNPCA, testFeaturesNPCA)


if __name__ == '__main__':
    main()
