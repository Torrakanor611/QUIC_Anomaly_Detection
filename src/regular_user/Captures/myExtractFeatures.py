import argparse, os
import scipy.stats as stats
import numpy as np

   
def extractStats(data):
    nSamp,nCols=data.shape
    p=[75,90,95,98]
    M1 = M2 = Md1 = Md2 = Std1 = Std2 = S1 = S2 = Pr1 = Pr2 = None
    
    indx1 = np.any(np.where(data[:,1] < 1))
    indx2 = np.any(np.where(data[:,1] >= 1))
    mystack = list()
    if np.any(indx1):
        dataL1 = data[indx1][0]
        #print(dataL1)
        M1=np.mean(dataL1,axis=0)
        Md1=np.median(dataL1,axis=0)
        Std1=np.std(dataL1,axis=0)
        S1=stats.skew(dataL1)
        #K1=stats.kurtosis(data)
        Pr1=np.array(np.percentile(dataL1,p,axis=0)).T.flatten()
        #print("< 1 features \n")
        #print(M1)
        mystack.append(M1)
        mystack.append(Md1)
        mystack.append(Std1)
        #mystack.append(S1)
        mystack.append(Pr1)        
    
    elif np.any(indx2):
        dataG1 = data[indx2][0]
        # print(dataG1)    
        M2=np.mean(dataG1,axis=0)
        Md2=np.median(dataG1,axis=0)
        Std2=np.std(dataG1,axis=0)
        S2=stats.skew(dataG1)
        #K1=stats.kurtosis(data)
        Pr2=np.array(np.percentile(dataG1,p,axis=0)).T.flatten()
        # print("> 1 features \n")
        # print(M2)
        mystack.append(M2)
        mystack.append(Md2)
        mystack.append(Std2)
        #mystack.append(S2)
        mystack.append(Pr2)      

    
    ## All features
    print("\n\nAll features")
    features=np.hstack(mystack)
    print(features)

    ## TODO: contagem de pacotes c menos 1 sec pra anterior => periodo cada janela
    ## TODO: perceber pq skew dá nan as vezes => buga os resultados
    return(features)

def extractFeatures(dirname,basename,nObs,allwidths):
    for o in range(0,nObs):
        features=np.array([])
        for oW in allwidths:
            obsfilename=dirname+"/"+basename+str(o)+"_w"+str(oW)+".dat"
            subdata=np.loadtxt(obsfilename)[:,1:]    #Loads data and removes first column (sample index)
            faux=extractStats(subdata)
            features=np.hstack((features,faux))
            
            #faux2=extractStatsSilenceActivity(subdata)
            #features=np.hstack((features,faux2))

        if o==0:
            obsFeatures=features
        else:
            obsFeatures=np.vstack((obsFeatures,features))

    return obsFeatures

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input dir')
    parser.add_argument('-w', '--widths', nargs='*',required=True, help='list of observation windows widths')
    parser.add_argument('-o', '--output', nargs='?',required=False, help='output file')
    args=parser.parse_args()
    
    dirname=args.input
    if isinstance(args.widths, list):
        allwidths=args.widths
    else:
        allwidths=list(args.widths)
    
    allfiles=os.listdir(dirname)
    nObs=len([f for f in allfiles if '_w{}.'.format(allwidths[0]) in f])
    lbn=allfiles[0].rfind("obs")+3
    basename=allfiles[0][:lbn]
    
    #print("nº observaçoes: ",nObs)
    
    features=extractFeatures(dirname,basename,nObs,allwidths)
    
    if args.output is None:
        outfilename=basename+"_features.dat"
    else:
        outfilename=args.output
    
    np.savetxt(outfilename,features,fmt='%d')
    
    print(features.shape)
        

if __name__ == '__main__':
    main()
