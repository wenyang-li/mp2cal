import sys, numpy as np
import pyuvdata.uvdata as uvd
obs = sys.argv[1]
inpath = obs+'.uvfits'
outpath = './raw_data/'+obs+'.uvfits'
autos = np.load(obs+'_auto.npz')
auto = {}
for k in autos.keys():
    if k.isdigit():
        auto[int(k)] = autos[k]
uv = uvd.UVData()
uv.read_uvfits(inpath, run_check=False)
a1 = uv.ant_1_array[:uv.Nbls]
a2 = uv.ant_2_array[:uv.Nbls]
data = uv.data_array.reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs,uv.Npols)
for ii in range(uv.Nbls):
    data[:,ii,:,0] *= (auto[a1[ii]][:,0]*auto[a2[ii]][:,0])
    data[:,ii,:,1] *= (auto[a1[ii]][:,1]*auto[a2[ii]][:,1])
uv.data_array = data.reshape(uv.Nblts,uv.Nspws,uv.Nfreqs,uv.Npols)
uv.write_uvfits(outpath)

