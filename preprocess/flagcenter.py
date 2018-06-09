import sys, numpy as np, subprocess
import pyuvdata.uvdata as uvd
obs = sys.argv[1]
uv = uvd.UVData()
fchan = []
fname = obs+'.uvfits'
uv.read_uvfits(fname)
for ii in range(uv.Nfreqs):
    if ii%32 == 16: fchan.append(ii)
copyfile = obs+'_copy.uvfits'
subprocess.call(['mv',fname,copyfile])
uv.flag_array[:,:,fchan,:] = True
uv.write_uvfits(fname)
subprocess.call(['rm',copyfile])
