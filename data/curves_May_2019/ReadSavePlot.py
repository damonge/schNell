# salvatore vitale 2016

import scipy.io as sio
import numpy as np

USAGE="""

Read in a matlab file with PSD curves and save an ascii file for each required PSD
All psds lowercase by default.

python readMat_save.py -m matlab.mab o1 aligo

If not sure what psds are available do

python readMat_save.py -m matlab.mab --show_psds

not providing a psd list will result in all available psds to be saved

"""

if __name__=='__main__':

    from optparse import OptionParser
    parser=OptionParser(USAGE)
    parser.add_option("-o","--outpath", dest="outpath",default='.', help="store ascii in DIR (default CWD)", metavar="DIR")
    parser.add_option("-m","--matfile", dest="matfile", help="Matlab file containing curves", metavar="file.mat",default=None)
    parser.add_option("-p","--show_psds",dest='show',action="store_true",help='Print known PSDs and exit',default=False)
    parser.add_option("--dont_plot",dest='dontplot',action="store_true",help="Won't plot selected PSDs",default=False)

    (opts,args)=parser.parse_args()

    
    required_psds= args
    npsd=len(args)
    allpsd=False
    if len(args)==0:
        allpsd=True

    mfile=opts.matfile
    import os
    import sys
    dontplot=opts.dontplot    
    if opts.outpath is not '.':
        if not os.path.isdir(opts.outpath):
            print("Directory %s does not exist or is not writeable"%opts.outpath)
            sys.exit(1)
    if opts.matfile is None:
        print("Must provide matlab file")
        sys.exit(1)
    if not os.path.isfile(mfile):
        print("cannot find file %s. Exiting\n"%mfile)
        exit(1)
    allNoiseFile=sio.loadmat(mfile)
    allNoises=allNoiseFile['curves']

    if opts.show is True:
        tmp=allNoises[0,0].dtype
        print("Known curves are: \n")
        for i in tmp.fields:
            print(i.lower())
        sys.exit(1)
    if allpsd is False:
        tmp=allNoises[0,0].dtype
        for p in args:
            if not p.lower() in [i.lower() for i in list(tmp.fields)]:
                print("PSD %s is not in file %s\n"%(p, opts.matfile))
                print("Known curves are:\n")
                for i in tmp.fields:
                    print(i.lower())

    if opts.dontplot is False:
        import pylab
        myfig = pylab.figure("Noise curves")
        from itertools import cycle
        lines = ["-","--","-.",":"]
        linecycler = cycle(lines) 

    nfprefix="curve"
    nfpostfix=".dat"
    for (idx,name) in zip(range(len(allNoises[0,0])),list((allNoises[0,0].dtype).names)):
        if allpsd is False and not name.lower() in args:
            continue
        
        fvsnoise=allNoises[0,0][idx].T
        if (len(fvsnoise)>1):
                freq=fvsnoise[0]
                noise=fvsnoise[1]
                nfprefix=name.lower()
                savename=os.path.join(opts.outpath,nfprefix+nfpostfix)
                np.savetxt(savename,np.transpose([freq,noise]),fmt='%.7e')
                if dontplot is False:
                    pylab.loglog(freq,noise,next(linecycler),label="%s"%name)

if dontplot is False:
    pylab.xlabel('Frequency [Hz]')
    pylab.ylabel('Strain [1/Hz$^{1/2}$]')
    pylab.legend(loc='best')
    try:
        myfig.savefig("Noises.pdf")
    except:
        myfig.savefig("Noises.png")
    pylab.show()
    pylab.clf()
