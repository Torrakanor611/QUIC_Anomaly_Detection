import argparse, os
import scipy.stats as stats
import numpy as np

   
def extractStats(data):
    nSamp,nCols=data.shape
    # print(data)
    pkt_len, time = np.hsplit(data, 2)
    # print(pkt_len)
    # print(time)
    times_m1seg = time[time < 1].flatten()
    times_M1seg = time[time >= 1].flatten()
    print(times_M1seg)
    print(times_m1seg)

    # (time independent) packet length features
    mean_pl = np.mean(pkt_len)
    median_pl = np.median(pkt_len)
    StdDev_pl=np.std(pkt_len)
    #Skew_pl=stats.skew(pkt_len)[0]
    p=[95]
    perc_pl=np.array(np.percentile(pkt_len,p)).T.flatten()[0]
    
    # print("mean_pl", mean_pl) 
    # print("median_pl", median_pl)
    # print("StdDev_pl", StdDev_pl)
    # print("Skew_pl", Skew_pl)
    # print("perc_pl", perc_pl)

    mean_m1seg_t = median_m1seg_t = StdDev_m1seg_t = count_m1seg_t = 0
    mean_M1seg_t = median_M1seg_t = StdDev_M1seg_t = count_M1seg_t = 0

    # (time dependent) time for previous packet features
    # < 1seg
    if not times_m1seg.size == 0:
        mean_m1seg_t = np.mean(times_m1seg)
        median_m1seg_t = np.median(times_m1seg)
        StdDev_m1seg_t = np.std(times_m1seg)
        count_m1seg_t = len(times_m1seg)

    # > 1seg
    if not times_M1seg.size == 0:
        mean_M1seg_t = np.mean(times_M1seg)
        median_M1seg_t = np.median(times_M1seg)
        StdDev_M1seg_t = np.std(times_M1seg)
        count_M1seg_t = len(times_M1seg)

    # print("mean_m1seg_t", mean_m1seg_t)
    # print("median_m1seg_t", median_m1seg_t)
    # print("StdDev_m1seg_t", StdDev_m1seg_t)
    # print("mean_M1seg_t", mean_M1seg_t)
    # print("Median_M1seg_t", median_M1seg_t)
    # print("StdDev_M1seg_t", StdDev_M1seg_t)

    features=np.hstack((mean_pl, median_pl, StdDev_pl, perc_pl, mean_m1seg_t, \
        mean_M1seg_t, median_m1seg_t, median_M1seg_t, StdDev_m1seg_t, StdDev_M1seg_t, \
        count_m1seg_t, count_M1seg_t))
    print(features)
    
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
