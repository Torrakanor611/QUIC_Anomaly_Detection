import sys
import argparse
import datetime
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark
import statistics
import numpy as np

def pktHandler(timestamp,srcIP,dstIP,lengthIP,sampDelta,outfile, protocol, srcPort, dstPort):
    global scnets
    global ssnets
    global npkts
    global T0
    global outc
    global last_ks
    # added our metrics
    global lastTimestamp
    global auxPktLen # list of pkt Len
    auxPktLen = []
    
    if (IPAddress(srcIP) in scnets and IPAddress(dstIP) in ssnets) or (IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):
        if npkts==0:
            T0=float(timestamp)
            last_ks=0
            minPktLen = 10000 # high value
            maxPktLen = -1 # low value

            
        ks=int((float(timestamp)-T0)/sampDelta)
        # {number} {num.of pkt cli -> serv} {bytes upload} {num.of pkt serv-> cli} {bytes download} \
        # {total num.of pkt in the flow} {max pkt len} {min pkt len} {average pkt len} {median pkt len} \
        # {mode pkt len} {strd dev pkt len} {95th perc}
        if ks>last_ks:
            outfile.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(last_ks,*outc))
            print('{} {} {} {} {} {} {} {} {} {} {} {} {}'.format(last_ks,*outc))
            outc=[0,0,0,0,0,0,0,0,0,0,0,0,0]  
            
        if ks>last_ks+1:
            for j in range(last_ks+1,ks):
                outfile.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(j,*outc))
                print('{} {} {} {} {} {} {} {} {} {} {} {} {}'.format(j,*outc))

        # partir por clientes, depois gerar
        # o que tem mais o tamanho do pacote cli -> serv
        # numero de pacotes
        # tempo para o pacote anterior 
        # toda a infromação para mais tar
        # manter a variancia e a média


        if IPAddress(dstIP) in scnets:  # Download / cli -> serv
            outc[2]=outc[2]+1
            outc[3]=outc[3]+int(lengthIP)

            auxPktLen = auxPktLen + int(lengthIP)

            if int(lengthIP) > maxPktLen:
                maxPktLen = int(lengthIP)
            
            if int(lengthIP) < minPktLen:
                minPktLen = int(lengthIP)

            outc[7] = statistics.mean(auxPktLen)
            strdDevPktLen = statistics.stdev(auxPktLen)


        outc[5] = maxPktLen
        outc[6] = minPktLen
        outc[8] = medPktLen
        outc[9] = modePktLen
        outc[10] = strdDevPktLen
        outc[11] = _95thPecPktLen

        npkts=npkts+1
        last_ks=ks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-o', '--output', nargs='?',required=False, help='output file')
    parser.add_argument('-f', '--format', nargs='?',required=True, help='format',default=1)
    parser.add_argument('-d', '--delta', nargs='?',required=False, help='samplig delta interval')
    parser.add_argument('-c', '--cnet', nargs='+',required=True, help='client network(s)')
    parser.add_argument('-s', '--snet', nargs='+',required=True, help='service network(s)')
    
    args=parser.parse_args()
    
    if args.delta is None:
        sampDelta=1
    else:
        sampDelta=float(args.delta)
    
    cnets=[]
    for n in args.cnet:
        try:
            nn=IPNetwork(n)
            cnets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    #print(cnets)
    if len(cnets)==0:
        print("No valid client network prefixes.")
        sys.exit()
    global scnets
    scnets=IPSet(cnets)

    snets=[]
    for n in args.snet:
        try:
            nn=IPNetwork(n)
            snets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    #print(snets)
    if len(snets)==0:
        print("No valid service network prefixes.")
        sys.exit()
        
    global ssnets
    ssnets=IPSet(snets)
        
    fileInput=args.input
    fileFormat=int(args.format)
    
    if args.output is None:
        fileOutput=fileInput+"_d"+str(sampDelta)+".dat"
    else:
        fileOutput=args.output
        
    global npkts
    global T0
    global outc
    global last_ks
    global auxPktLen # list of pkt Len

    npkts=0
    outc=[0,0,0,0]
    #print('Sampling interval: {} second'.format(sampDelta))

    outfile = open(fileOutput,'w') 
    capture = pyshark.FileCapture(fileInput,display_filter='ip')
    for pkt in capture:
        protocol =  pkt.transport_layer
        timestamp= pkt.sniff_timestamp
        srcIP= pkt.ip.src
        dstIP= pkt.ip.dst
        lengthIP= pkt.ip.len
        srcPort= protocol.srcport
        dstPort=protocol.dstport
        ## added src and dst udp port
        pktHandler(timestamp,srcIP,dstIP,lengthIP,sampDelta,outfile, protocol, srcPort, dstPort)
    
    outfile.close()

if __name__ == '__main__':
    main()
