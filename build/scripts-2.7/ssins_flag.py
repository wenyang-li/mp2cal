from matplotlib import use
use('Agg')
import numpy as np, mp2cal, sys, optparse, subprocess
import pyuvdata.uvdata as uvd

o = optparse.OptionParser()
o.set_usage('ssins_flag.py [options] obsid')
o.set_description(__doc__)
o.add_option('-i',dest='inpath',default='./',help='path to input uvfits')
o.add_option('-o',dest='outpath',default='./',help='path to output uvfits')
opts,args = o.parse_args(sys.argv[1:])
obs = args[0]
uv = uvd.UVData()
uv.read_uvfits(opts.inpath+obs+'.uvfits')
uv.Npols = 2
uv.flag_array = uv.flag_array[:,:,:,:2]
uv.data_array = uv.data_array[:,:,:,:2]
uv.nsample_array = uv.nsample_array[:,:,:,:2]
uv.polarization_array = uv.polarization_array[:2]
if uv.Nfreqs == 768:
    for ii in range(768):
        if ii%32==16: uv.flag_array[:,:,ii,:] = True
ins = mp2cal.qltm.INS(uv)
ins.outliers_flagging()
ins.time_flagging()
ins.coherence_flagging()
ins.outliers_flagging()
ins.merge_flagging()
ins.freq_flagging()
ins.apply_flagging()
ins.saveplots(opts.outpath, obs.split('/')[-1])
ins.savearrs(opts.outpath, obs.split('/')[-1])
ins.uv.write_uvfits(opts.outpath+obs+'.uvfits')
