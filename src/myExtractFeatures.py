import argparse, os
import scipy.stats as stats
import numpy as np


def extractStats(data):
    pkt_len, time = np.hsplit(data, 2)
    times_m1seg = time[time < 1].flatten()
    times_M1seg = time[time >= 1].flatten()
    # (time independent) packet length features
    mean_pl = np.mean(pkt_len)
    median_pl = np.median(pkt_len)
    StdDev_pl = np.std(pkt_len)
    p=[95]
    perc_pl = np.array(np.percentile(pkt_len,p)).T.flatten()[0]
    #mod_pl = stats.mode(pkt_len)
    max_pl = np.max(pkt_len)
    min_pl = np.min(pkt_len)

    mean_m1seg_t = median_m1seg_t = StdDev_m1seg_t = count_m1seg_t = 0
    mean_M1seg_t = median_M1seg_t = StdDev_M1seg_t = count_M1seg_t = 0

    # (time dependent) time for previous packet features
    # tim < 1seg
    if not times_m1seg.size == 0:
        mean_m1seg_t = np.mean(times_m1seg)
        median_m1seg_t = np.median(times_m1seg)
        StdDev_m1seg_t = np.std(times_m1seg)
        count_m1seg_t = len(times_m1seg)

    # time > 1seg
    if not times_M1seg.size == 0:
        mean_M1seg_t = np.mean(times_M1seg)
        median_M1seg_t = np.median(times_M1seg)
        StdDev_M1seg_t = np.std(times_M1seg)
        count_M1seg_t = len(times_M1seg)

    features=np.hstack((mean_pl, median_pl, StdDev_pl, perc_pl, mean_m1seg_t, \
        mean_M1seg_t, median_m1seg_t, median_M1seg_t, StdDev_m1seg_t, min_pl, \
        max_pl, StdDev_M1seg_t, count_m1seg_t, count_M1seg_t))

    return(features)

def extractFeatures(dirname,basename,nObs,allwidths):
    for o in range(0,nObs):
        features=np.array([])
        for oW in allwidths:
            obsfilename=dirname+"/"+basename+str(o)+"_w"+str(oW)+".dat"
            subdata=np.loadtxt(obsfilename)[:,1:]    #Loads data and removes first column (sample index)
            faux=extractStats(subdata)
            features=np.hstack((features,faux))
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

    features=extractFeatures(dirname,basename,nObs,allwidths)

    if args.output is None:
        outfilename=basename+"_features.dat"
    else:
        outfilename=args.output

    np.savetxt(outfilename,features,fmt='%1.3e')

    print(features.shape)


if __name__ == '__main__':
    main()