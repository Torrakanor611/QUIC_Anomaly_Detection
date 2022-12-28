import argparse, os
import numpy as np
        
def slidingObsWindow(data,lengthObsWindow,slidingValue,basename):
    iobs=0
    nSamples,nMetrics=data.shape
    obsData=np.zeros((0,lengthObsWindow,nMetrics))
    for s in np.arange(lengthObsWindow,nSamples,slidingValue):
        subdata=data[s-lengthObsWindow:s,:]
        obsData=np.insert(obsData,iobs,subdata,axis=0)
        
        obsFname="{}_obs{}_w{}.dat".format(basename,iobs,lengthObsWindow)
        iobs+=1
        np.savetxt(obsFname,subdata,fmt='%1.8e')
               
    return obsData # 3D arrays (obs, sample, metric)
        

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-w', '--widths', nargs='*',required=False, help='list of observation windows widths',default=60)
    parser.add_argument('-s', '--slide', nargs='?',required=False, help='observation windows slide value',default=10)
    args=parser.parse_args()
    #janelas de observação 20 seg, sliding de 5 segundos
    ## python3 myObsWindows.py -i .pcap -w 20 -s 5
    
    #Argument handling
    fileInput=args.input
    lengthObsWindow=[int(w) for w in args.widths]
    slidingValue=int(args.slide)
        
    ## Loading samples data
    data=np.loadtxt(fileInput,dtype=float)

    ## Creating directory of observations
    fname=''.join(fileInput.split('.')[:-1])
    dirname=fname+"_obs_s{}".format(slidingValue)
    os.mkdir(dirname)

    print("\n\n### SLIDING Observation Windows with Length {} and Sliding {} ###".format(lengthObsWindow[0],slidingValue))
    obsData=slidingObsWindow(data,lengthObsWindow[0],slidingValue,dirname+"/"+fname)
    print(obsData)

            
if __name__ == '__main__':
    main()
