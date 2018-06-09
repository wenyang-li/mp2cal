import numpy as np, sys, os, gc, optparse
import pyuvdata.uvdata as uvd
from cnn import *
o = optparse.OptionParser()
o.set_usage('cotter_fq_ave.py [options] obsid')
o.set_description(__doc__)
o.add_option('-i',dest='inpath',default='./',help='path to input uvfits')
o.add_option('-o',dest='outpath',default='./data/',help='path to output uvfits')
o.add_option('--rewrite',dest='rewrite',default=False, action='store_true', help='rewrite uvfits, with extra flagging')
o.add_option('--xpol',dest='xpol',default=False, action='store_true', help='drop out cross pol')
o.add_option('--ins',dest='ins',default='./mwilensky/',help='path to ins output')
o.add_option('--flgtype',dest='flgtype',default='Unflagged',help='choose All or Unflagged')
opts,args = o.parse_args(sys.argv[1:])
obs = args[0]
uv = uvd.UVData()
print 'loading...'
fname = opts.inpath+obs+'.uvfits'
fout = opts.outpath+obs+'.uvfits'
flgtype = opts.flgtype
uv.read_uvfits(fname,run_check=False,run_check_acceptability=False)
INS = np.load(opts.ins+flgtype+'/all_spw/arrs/'+obs+'_'+flgtype+'_Amp_INS_mask.npym')
time_flagging(INS)
freq_flagging(INS)
coherence_flagging(INS)
extend_flagging(INS)
if opts.xpol:
    uv.Npols = 2
    uv.flag_array = uv.flag_array[:,:,:,:2]
    uv.data_array = uv.data_array[:,:,:,:2]
    uv.nsample_array = uv.nsample_array[:,:,:,:2]
    uv.polarization_array = uv.polarization_array[:2]
if opts.rewrite:
    fout = fname
    import subprocess
    np.save(opts.inpath+obs+'_rawflg.npy',uv.flag_array)
    further_flagging(uv, INS[:,:,:,:uv.Npols])
    fcopy = '.'.join(fout.split('.')[:-1])+'_copy.uvfits'
    subprocess.call(['mv',fout,fcopy])
else:
    further_flagging(uv, INS[:,:,:,:uv.Npols])
    if not os.path.exists(opts.outpath): os.makedirs(opts.outpath)
print 'writing...'
uv.write_uvfits(fout,spoof_nonessential=True)
if opts.rewrite: subprocess.call(['rm',fcopy])
