import numpy as np, pyuvdata.uvdata as uvd
import sys, subprocess

def get_auto(uv):
    a1 = uv.ant_1_array[:uv.Nbls]
    a2 = uv.ant_2_array[:uv.Nbls]
    auto = {}
    ind = np.where(a1 == a2)[0]
    inx = np.where(a1 != a2)[0]
    data = uv.data_array.reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs,uv.Npols)
    wgts = np.logical_not(uv.flag_array.reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs,uv.Npols))
    wgts = (np.sum(wgts[:,inx,:,:], axis=1) >0)
    wgts = np.sum(wgts, axis=1)
    wxmax = np.max(wgts[:,0])
    wymax = np.max(wgts[:,1])
    tindx = np.where(wgts[:,0] == wxmax)[0]
    tindy = np.where(wgts[:,1] == wymax)[0]
    print "Ntime samples:", tindx.size, tindy.size
    tile_order = []
    for ii in ind:
        a = a1[ii]
        tile_order.append(a)
        auto[a] = np.zeros((uv.Nfreqs,2))
        auto[a][:,0] = np.sqrt(np.mean(np.abs(data[tindx,ii,:,0]),axis=0))
        auto[a][:,1] = np.sqrt(np.mean(np.abs(data[tindy,ii,:,1]),axis=0))
    while True:
        aref = tile_order[0]
        if np.any(auto[aref]==0):
            tile_order.remove(aref)
            continue
        else:
            amp_ref = np.copy(auto[aref])
            break
    for a in auto.keys():
        auto[a] /= amp_ref
        ind0 = np.where(auto[a]==0)
        auto[a][ind0] += 1e-8
    print "ref antenna: ", aref
    return auto, aref

def save_auto(auto, aref, obs):
    acopy = {}
    for a in auto.keys(): acopy[str(a)] = auto[a]
    acopy['ref'] = np.array(aref)
    np.savez(obs+"_auto.npz",**acopy)

def divide_auto(uv, auto):
    a1 = uv.ant_1_array[:uv.Nbls]
    a2 = uv.ant_2_array[:uv.Nbls]
    data = uv.data_array.reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs,uv.Npols)
    for ii in range(uv.Nbls):
        data[:,ii,:,0] /= (auto[a1[ii]][:,0]*auto[a2[ii]][:,0])
        data[:,ii,:,1] /= (auto[a1[ii]][:,1]*auto[a2[ii]][:,1])
        #data[:,ii,:,2] /= (auto[a1[ii]][:,:,0]*auto[a2[ii]][:,:,1])
        #data[:,ii,:,3] /= (auto[a1[ii]][:,:,1]*auto[a2[ii]][:,:,0])
    uv.data_array = data.reshape(uv.Nblts,uv.Nspws,uv.Nfreqs,uv.Npols)

obs = sys.argv[1]
uv = uvd.UVData()
fname = obs + '.uvfits'
uv.read_uvfits(fname)
auto, aref = get_auto(uv)
save_auto(auto, aref, obs)
subprocess.call(['mv',fname,obs+'_copy.uvfits'])
if uv.Npols == 4:
    uv.Npols = 2
    uv.flag_array = uv.flag_array[:,:,:,:2]
    uv.data_array = uv.data_array[:,:,:,:2]
    uv.nsample_array = uv.nsample_array[:,:,:,:2]
    uv.polarization_array = uv.polarization_array[:2]
divide_auto(uv, auto)
uv.write_uvfits(fname)
subprocess.call(['rm',obs+'_copy.uvfits'])

