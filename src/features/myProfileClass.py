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
    waitforEnter(False)

     
def outputStatistics(method, tp, fp, tn, fn):
    ac = ((tp+tn)/(tp+tn+fp+fn))*100
    pr = (tp/(tp+fp))*100
    rec = (tp/(tp+fn))*100
    sc = 0
    if (rec+pr) == 0:
        sc = 0
    else:
        sc = (2*(rec*pr))/(rec+pr)

    print("-- Analysis of "+method+" --")
    print("True Positives: {:>4}, False Negatives: {:>4}".format(tp, fn))
    print("True Negatives: {:>4}, False Positives: {:>4}".format(tn, fp))
    print("")
    print("{:<37}{:>6}".format("Accuracy (%): ", str(round(ac, 2))))
    print("{:<37}{:>6}".format("Precision (%): ", str(round(pr, 2))))
    print("{:<37}{:>6}".format("Recall (%): ", str(round(rec, 2))))
    print("{:<37}{:>6}".format("F1-Score: ", str(round(sc, 2))))
    print("-------------------------------------------\n")


def myPCA(trainFeaturesN,testFeaturesN, trainClass):
    pca = PCA(n_components=3, svd_solver='full')

    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)
    testFeaturesNPCA = trainPCA.transform(testFeaturesN)

    return trainFeaturesNPCA,testFeaturesNPCA


def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))


def CentroidsDistance(trainClass, trainFeaturesN,testFeaturesN): 
    centroids={}
    pClass=(trainClass==0).flatten()
    centroids.update({0:np.mean(trainFeaturesN[pClass,:],axis=0)})
    print('All Features Centroids:\n',centroids)
    pos = neg = 0
    AnomalyThreshold=1.4
    print('\n-- Anomaly Detection based on Centroids Distances --')
    nObsTest,nFea=testFeaturesN.shape
    for i in range(nObsTest):
        x=testFeaturesN[i]
        dist=distance(x,centroids[0])
        if dist >AnomalyThreshold:
            result="Anomaly"
            pos = pos + 1
        else:
            result="OK"
            neg = neg + 1

        print('Obs: {:2} ({}): Normalized Distance to Centroids: {:.4f}   -> Result -> {}'.format(i,Classes[oClass[i][0]],dist,result))
    return (pos, neg)


def CentroidsDistance_PCA(trainClass,trainFeaturesNPCA,testFeaturesNPCA):
    centroids={}
    pClass=(trainClass==0).flatten()
    centroids.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
    print('All Features Centroids:\n',centroids)

    pos = neg = 0
    AnomalyThreshold=1.4
    print('\n-- Anomaly Detection based on Centroids Distances (PCA Features) --')
    nObsTest,nFea=testFeaturesNPCA.shape
    for i in range(nObsTest):
        x=testFeaturesNPCA[i]
        dist=distance(x,centroids[0])
        if dist > AnomalyThreshold:
            result="Anomaly"
            pos = pos + 1
        else:
            result="OK"
            neg = neg + 1
        
        print('Obs: {:2} ({}): Normalized Distance to Centroids (PCA): {:.4f} -> Result -> {}'.format(i,Classes[oClass[i][0]],dist,result))
    return (pos, neg)


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

    pos = neg = 0
    AnomalyThreshold=0.10
    nObsTest,nFea=testFeaturesNPCA.shape
    for i in range(nObsTest):
        x=testFeaturesNPCA[i,:]
        prob = multivariate_normal.pdf(x,means[0],covs[0])
        if prob < AnomalyThreshold:
            result="Anomaly"
            pos = pos + 1
        else:
            result="OK"
            neg = neg + 1
        
        print('Obs: {:2} ({}): Probabilities: {:.4e} -> Result -> {}'.format(i,Classes[oClass[i][0]],prob,result))
    return (pos, neg)


def OneClassSVM_PCA(trainFeaturesNPCA, testFeaturesNPCA):
    print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
    ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesNPCA)  
    rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesNPCA)  
    poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesNPCA)  

    L1=ocsvm.predict(testFeaturesNPCA)
    L2=rbf_ocsvm.predict(testFeaturesNPCA)
    L3=poly_ocsvm.predict(testFeaturesNPCA)

    AnomResults={-1:"Anomaly",1:"OK"}

    linpos = linneg = RBFpos = RBFneg = polypos = polyneg = 0
    nObsTest,nFea=testFeaturesNPCA.shape
    for i in range(nObsTest):
        linpos = linpos + 1 if L1[i] == -1 else linpos
        linneg = linneg + 1 if L1[i] == 1 else linneg
        RBFpos = RBFpos + 1 if L2[i] == -1 else RBFpos
        RBFneg = RBFneg + 1 if L2[i] == 1 else RBFneg
        polypos = polypos + 1 if L3[i] == -1 else polypos
        polyneg = polyneg + 1 if L3[i] == 1 else polyneg
        print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[oClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    return (linpos, linneg, RBFpos, RBFneg, polypos, polyneg)


def OneClassSVM(trainFeaturesN, testFeaturesN):
    print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
    ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesN)  
    rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesN)  
    poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesN)  

    L1=ocsvm.predict(testFeaturesN)
    L2=rbf_ocsvm.predict(testFeaturesN)
    L3=poly_ocsvm.predict(testFeaturesN)

    AnomResults={-1:"Anomaly",1:"OK"}

    linpos = linneg = RBFpos = RBFneg = polypos = polyneg = 0
    nObsTest,nFea=testFeaturesN.shape
    for i in range(nObsTest):
        linpos = linpos + 1 if L1[i] == -1 else linpos
        linneg = linneg + 1 if L1[i] == 1 else linneg
        RBFpos = RBFpos + 1 if L2[i] == -1 else RBFpos
        RBFneg = RBFneg + 1 if L2[i] == 1 else RBFneg
        polypos = polypos + 1 if L3[i] == -1 else polypos
        polyneg = polyneg + 1 if L3[i] == 1 else polyneg
        print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[oClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    return (linpos, linneg, RBFpos, RBFneg, polypos, polyneg)


def _percentage(arr):
    aux = [0, 0, 0, 0]
    aux[0] = round((float(arr[0]) / (arr[0] + arr[1]))*100)
    aux[1] = round((float(arr[1]) / (arr[0] + arr[1]))*100)
    aux[2] = round((float(arr[2]) / (arr[2] + arr[3]))*100)
    aux[3] = round((float(arr[3]) / (arr[2] + arr[3]))*100)
    return aux


def main():
    plt.ion()
    ## FEATURE EXTRACTION
    ## -- Load features from browsing and attack -- ##
    #dirname = "ObsWind_10s_Slide_2s/"
    dirname = "ObsWind_20s_Slide_5s/"
    #dirname = "ObsWind_60s_Slide_10s/"

    features_browsing = np.loadtxt(dirname+"QUIC_45minspcap_np-29385_obs_features.dat")
    features_browsing1 = np.loadtxt(dirname+"QUIC_26minspcap_np-3604_obs_features.dat")
    features_browsing2 = np.loadtxt(dirname+"QUIC_25mins_Manualpcap_np-17588_obs_features.dat")

    features_browsing = np.vstack((features_browsing, features_browsing1, features_browsing2))
    oClass_browsing=np.ones((len(features_browsing),1))*0

    features_dos = np.loadtxt(dirname+"3_dummy-np-20000_obs_features.dat")
    features_dos1 = np.loadtxt(dirname+"3_dummy1-np-20000_obs_features.dat")
    features_dos2 = np.loadtxt(dirname+"3_smarter-np-20000_obs_features.dat")

    features_dos = np.vstack((features_dos, features_dos1, features_dos2))
    oClass_dos=np.ones((len(features_dos),1))*0

    features=np.vstack((features_browsing,features_dos))
    global oClass 
    oClass = np.vstack((oClass_browsing,oClass_dos))
    print('Train Stats Features Size:',features.shape)

    global nfig
    nfig = 1
    nfig += 1

    percentage=0.5

    ## FEATURE DATASETS FOR TRAINING AND TEST

    pB=int(len(features_browsing)*percentage)   ## Train with licit traffic (Browsing)
    trainFeatures_browsing=features_browsing[:pB,:]
    trainFeatures=np.vstack((trainFeatures_browsing))
    trainClass=np.vstack((oClass_browsing[:pB],))

    pD=int(len(features_dos)*percentage)        ## Test with mixed traffic (Browsing+DoS)
    testFeatures_dos=features_dos[pD:,:]
    # Num. of entrys from dos dataset
    nObs_dos, _ = testFeatures_dos.shape

    testFeatures_browsing=features_browsing[pB:,:]
    # Num. of entrys from browsing dataset
    nObs_b, _ = testFeatures_browsing.shape
    testFeatures=np.vstack((testFeatures_browsing,testFeatures_dos))

    # N. observations of testFeatures dataset
    nObs, _ = testFeatures.shape
    testClass=np.vstack((oClass_browsing[pB:],oClass_dos[pD:])) 


    ## FEATURE NORMALIZATION 
    trainScaler = MaxAbsScaler().fit(trainFeatures)
    trainFeaturesN=trainScaler.transform(trainFeatures)
    testFeaturesN=trainScaler.transform(testFeatures)


    ## FEATURE REDUCTION
    trainFeaturesNPCA,testFeaturesNPCA = myPCA(trainFeaturesN, testFeaturesN, trainClass)


    ## ANOMALY DETECTION Statistics
    AnomaliesCD ,_ = CentroidsDistance(trainClass, trainFeaturesN, testFeaturesN)
    AnomaliesCD_PCA ,_ = CentroidsDistance_PCA(trainClass, trainFeaturesNPCA, testFeaturesNPCA)
    AnomaliesMlt ,_ = MultivariatePDF_PCA(trainClass, trainFeaturesNPCA, testFeaturesNPCA)

    ## ANOMALY DETECTION Machine Learning
    AnomaliesL, _, AnomaliesRBF, _, AnomaliesP, _ = OneClassSVM(trainFeaturesN, testFeaturesN)
    AnomaliesL_PCA, _, AnomaliesRBF_PCA, _, AnomaliesP_PCA, _ = OneClassSVM_PCA(trainFeaturesNPCA, testFeaturesNPCA)

    ## Evaluation of Anomaly Detection Results
     # pos = equal as classified as 'anomaly'
     # neg = equal as classified as 'ok'
     # training set -> browsing
     # test set -> DoS
    testFeatures_dos=features_dos[:,:] # all DoS data available
    testFeatures_DoS=np.vstack((testFeatures_dos))
    testFeaturesN_DoS=trainScaler.transform(testFeatures_DoS)
    trainFeaturesNPCA_DoS,testFeaturesNPCA_DoS = myPCA(trainFeaturesN, testFeaturesN_DoS, trainClass)

    # [tp, fp,tn,fn]
    cd = [0, 0, 0 ,0]
    cdPCA = [0, 0, 0 ,0]
    MvPCA =[0, 0, 0 ,0]
    _svmRbf =[0, 0, 0 ,0]
    _svmP = [0, 0, 0 ,0]
    svmPCAL =[0, 0, 0 ,0]
    svmPCARbf =[0, 0, 0 ,0]
    svmPCAP = [0, 0, 0 ,0]
    _svmL = [0,0,0,0]

    cd[0], cd[1] = CentroidsDistance(trainClass, trainFeaturesN, testFeaturesN_DoS)
    cdPCA[0], cdPCA[1] = CentroidsDistance_PCA(trainClass, trainFeaturesNPCA_DoS, testFeaturesNPCA_DoS)
    MvPCA[0], MvPCA[1] = MultivariatePDF_PCA(trainClass, trainFeaturesNPCA_DoS, testFeaturesNPCA_DoS)

    _svmL[0], _svmL[1], _svmRbf[0], _svmRbf[1], _svmP[0], _svmP[1] = OneClassSVM(trainFeaturesN, testFeaturesN_DoS)
    svmPCAL[0], svmPCAL[1], svmPCARbf[0], svmPCARbf[1], svmPCAP[0], svmPCAP[1] = OneClassSVM_PCA(trainFeaturesNPCA_DoS, testFeaturesNPCA_DoS)

     # training set -> browsing
     # test set -> browsing, but diferent from above training set
    testFeatures_browsing=features_browsing[pB:,:]
    testFeatures_B=np.vstack((testFeatures_browsing))
    testFeaturesN_B=trainScaler.transform(testFeatures_B)
    trainFeaturesNPCA_B,testFeaturesNPCA_B = myPCA(trainFeaturesN, testFeaturesN_B, trainClass)

    cd[2], cd[3] = CentroidsDistance(trainClass, trainFeaturesN, testFeaturesN_B)
    cdPCA[2], cdPCA[3] = CentroidsDistance_PCA(trainClass, trainFeaturesNPCA_B, testFeaturesNPCA_B)
    MvPCA[2], MvPCA[3] = MultivariatePDF_PCA(trainClass, trainFeaturesNPCA_B, testFeaturesNPCA_B)

    _svmL[2], _svmL[3], _svmRbf[2], _svmRbf[3], _svmP[2], _svmP[3] = OneClassSVM(trainFeaturesN, testFeaturesN_B)
    svmPCAL[2], svmPCAL[3], svmPCARbf[2], svmPCARbf[3], svmPCAP[2], svmPCAP[3] = OneClassSVM_PCA(trainFeaturesNPCA_B, testFeaturesNPCA_B)

    print(cd, cdPCA, MvPCA, _svmL, _svmRbf, _svmP, svmPCAL, svmPCARbf, svmPCAP)

    # Anomaly Counter
    print("- Total of Anomalies for each method -")
    print("{:<32}{:>6}".format("Num. Observações test dataset: ", nObs))
    print("{:<32}{:>6}".format("Num. entrys from browsing: ", nObs_b))
    print("{:<32}{:>6}".format("Num. entrys from DoS: ", nObs_dos))
    r = round((float(nObs_dos)/nObs)*100, 1)
    print("{:<32}{:>6}".format("Percentage of DoS entrys: ", r))
    print("{:<32}{:>6}".format("Expected anomalys: ", round(r/100*nObs, 1)))
    print("\n")
    print("{:<32}{:>6}".format("Centroids Distance: ", str(AnomaliesCD)))
    print("{:<32}{:>6}".format("Centroids Distance PCA: ", str(AnomaliesCD_PCA)))
    print("{:<32}{:>6}".format("Multivariate PCA: ", str(AnomaliesMlt)))
    print("{:<32}{:>6}".format("SVM Linear: ", str(AnomaliesL)))
    print("{:<32}{:>6}".format("SVM Linear PCA: ", str(AnomaliesL_PCA)))
    print("{:<32}{:>6}".format("SVM RBF: ", str(AnomaliesRBF)))
    print("{:<32}{:>6}".format("SVM RBF PCA: ", str(AnomaliesRBF_PCA)))
    print("{:<32}{:>6}".format("SVM Poly: ", str(AnomaliesP)))
    print("{:<32}{:>6}".format("SVM Poly PCA: ", str(AnomaliesP_PCA)))
    print("--------------------------------------\n")

    ## Output Accuracy and Precision Stats
    outputStatistics("Centroids Distances",cd[0],cd[1],cd[2],cd[3])
    outputStatistics("Centroids Distances (PCA)",cdPCA[0],cdPCA[1],cdPCA[2],cdPCA[3])
    outputStatistics("Multivariate PDF (PCA)",MvPCA[0],MvPCA[1],MvPCA[2],MvPCA[3])
    outputStatistics("One Class SVM Linear",_svmL[0],_svmL[1],_svmL[2],_svmL[3])  
    outputStatistics("One Class SVM RBF",_svmRbf[0],_svmRbf[1],_svmRbf[2],_svmRbf[3])    
    outputStatistics("One Class SVM Poly",_svmP[0],_svmP[1],_svmP[2],_svmP[3])
    outputStatistics("One Class SVM Linear (PCA)",svmPCAL[0],svmPCAL[1],svmPCAL[2],svmPCAL[3])
    outputStatistics("One Class SVM RBF (PCA)",svmPCARbf[0],svmPCARbf[1],svmPCARbf[2],svmPCARbf[3])
    outputStatistics("One Class SVM poly (PCA)",svmPCAP[0],svmPCAP[1],svmPCAP[2],svmPCAP[3])

    metrics = ['True pos.', 'False neg.', 'False pos.', 'True neg.']
    bar_colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:blue']
    bar_labels = ['tp', 'fn', 'fp', 'tn']

    cd = _percentage(cd)
    cdPCA = _percentage(cdPCA)
    MvPCA = _percentage(MvPCA)
    _svmL = _percentage(_svmL)
    _svmRbf = _percentage(_svmRbf)
    _svmP = _percentage(_svmP)
    svmPCAL = _percentage(svmPCAL)
    svmPCARbf = _percentage(svmPCARbf)
    svmPCAP = _percentage(svmPCAP)

    fig, axs = plt.subplots(3, 3)
    custom_ylim = (0, 100)
    plt.setp(axs, ylim=custom_ylim)
    axs[0, 0].bar(metrics, cd, label=bar_labels, color=bar_colors)
    axs[0, 0].set_title("Centroids Distances")
    axs[0, 0].set(ylabel='percentage (%)')
    axs[0, 1].bar(metrics, cdPCA, label=bar_labels, color=bar_colors)
    axs[0, 1].set_title("Centroids Distances (PCA)")
    axs[0, 1].set(ylabel='percentage (%)')
    axs[0, 2].bar(metrics, MvPCA, label=bar_labels, color=bar_colors)
    axs[0, 2].set_title("Multivariate PDF (PCA)")
    axs[0, 2].set(ylabel='percentage (%)')

    axs[1, 0].bar(metrics, _svmL, label=bar_labels, color=bar_colors)
    axs[1, 0].set_title("One Class SVM Linear")
    axs[1, 0].set(ylabel='percentage (%)')
    axs[1, 1].bar(metrics, _svmRbf, label=bar_labels, color=bar_colors)
    axs[1, 1].set_title("One Class SVM RBF")
    axs[1, 1].set(ylabel='percentage (%)')
    axs[1, 2].bar(metrics, _svmP, label=bar_labels, color=bar_colors)
    axs[1, 2].set_title("One Class SVM poly")
    axs[1, 2].set(ylabel='percentage (%)')

    axs[2, 0].bar(metrics, svmPCAL, label=bar_labels, color=bar_colors)
    axs[2, 0].set_title("One Class SVM Linear (PCA)")
    axs[2, 0].set(ylabel='percentage (%)')
    axs[2, 1].bar(metrics, svmPCARbf, label=bar_labels, color=bar_colors)
    axs[2, 1].set_title("One Class SVM RBF (PCA)")
    axs[2, 1].set(ylabel='percentage (%)')
    axs[2, 2].bar(metrics, svmPCAP, label=bar_labels, color=bar_colors)
    axs[2, 2].set_title("One Class SVM poly (PCA)")
    axs[2, 2].set(ylabel='percentage (%)')
    fig.tight_layout()
    plt.show()
    waitforEnter(True)

if __name__ == '__main__':
    main()
