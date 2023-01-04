import sys,argparse, pyshark
from netaddr import IPNetwork, IPAddress, IPSet
import numpy as np

def pktHandler(timestamp,dstIP,lengthIP,outfile, np, verbose):
    global ssnets
    global outc
    global lastTimestamp
    
    if IPAddress(dstIP) in ssnets:  # Download / cli -> serv
        outc[0]=int(lengthIP)
        outc[1]=float(timestamp) - lastTimestamp
        lastTimestamp = float(timestamp)
            
        outfile.write('{} {} {}\n'.format(np,*outc))
        if verbose == str(1):
            print('{} {} {}'.format(np,*outc))
        
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', nargs='?', required=False, help='verbose output', default=0)
    parser.add_argument('-np', '--number', nargs='?', required=True, help='number of packets to consider')
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-o', '--output', nargs='?',required=False, help='output file')
    parser.add_argument('-s', '--snet', nargs='+',required=True, help='service network(s)')

    # python3 myPktSampling.py -v 1 -np 500 -i ../data/3_dummy.pcapng -s 192.168.56.100/30 -o ./attacks/3_dummy-np-500.dat
    # python3 myPktSampling.py -v 1 -np 400 -i ../data/3_dummy1.pcapng -s 192.168.56.100/30 -o ./attacks/3_dummy1-np-400.dat
    # python3 myPktSampling.py -v 1 -np 1500 -i ../data/3_smarter.pcapng -s 192.168.56.100/30 -o ./attacks/3_smarter-np-1500.dat


    args=parser.parse_args()

    snets=[]
    for n in args.snet:
        try:
            nn=IPNetwork(n)
            snets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))
    if len(snets)==0:
        print("No valid service network prefixes.")
        sys.exit()
        
    global ssnets
    ssnets=IPSet(snets)
        
    fileInput=args.input
    
    fileOutput=fileInput+"_np-"+args.number+".dat" if args.output is None else args.output

    global outc
    global lastTimestamp

    npkts=0
    outc=[0,0,0,0]

    outfile = open(fileOutput,'w')
    capture = []
    try:
        capture = pyshark.FileCapture(fileInput,display_filter='ip')
    except FileNotFoundError as e:
        print(e)
        exit(2)

    np = 0 # number of packets processed
    lastTimestamp = float(capture[0].sniff_timestamp)

    for pkt in capture:
        timestamp= pkt.sniff_timestamp
        dstIP= pkt.ip.dst
        lengthIP= pkt.ip.len

        # process 1st np packets
        if np < int(args.number) + 1:
            pktHandler(timestamp,dstIP,lengthIP,outfile, np, args.verbose)
            np += 1
        else:
            break
    
    outfile.close()

if __name__ == '__main__':
    main()
