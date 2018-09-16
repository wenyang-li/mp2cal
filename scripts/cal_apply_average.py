import numpy as np, pyuvdata.uvdata as uvd
import aipy, mp2cal, sys, os, optparse
from scipy.io.idl import readsav

### Options ###
o = optparse.OptionParser()
o.set_usage('cal_apply_average.py [options] obs')
o.set_description(__doc__)
o.add_option('--fhdpath', dest='fhdpath', default='', type='string',
             help='path to fhd dir for fhd output visibilities if ftype is fhd. Include final / in path.')
o.add_option('--omnipath',dest='omnipath',default='',type='string', help='Path to load omnical solution files. Include final / in path.')
o.add_option('--omniapp',dest='omniapp',default=False,action='store_true',
             help='Toggle: apply omnical solutions to hex tiles. Default=False')
o.add_option('--subtract',dest='subtract',default=False,action='store_true',
             help='Toggle: subtract model vis. Default=False')
o.add_option('--outpath', dest='outpath', default='', type='string',
             help='path to output calibrated data. Include final / in path.')
opts,args = o.parse_args(sys.argv[1:])


obsid = args[0]

# Getting cal solutions
suffix = 'AF'
fhdsav = opts.fhdpath + 'calibration/' + obsid + '_cal.sav'
if not os.path.exists(fhdsav): raise IOError('%s not found' %fhdsav)
print "Get FHD solutions ..."
gains = mp2cal.io.load_gains_fhd(fhdsav, raw=True)
gbp = mp2cal.io.load_fhd_global_bandpass(opts.fhdpath, obsid)
for p in gains.keys():
    for a in gains[p].keys():
        ind = np.where(gains[p][a]!=0)
        amp = np.mean(np.abs(gains[p][a][ind]))/np.mean(gbp[p][ind])
        gains[p][a] = amp*gbp[p]*np.exp(1j*np.angle(gains[p][a]))
exec('from PhaseII_cal import *')
if opts.omniapp:
    suffix = 'AOF'
    print "Get omnical solutions ..."
    omnixx = opts.omnipath + obsid + '.xx.omni.npz'
    omniyy = opts.omnipath + obsid + '.yy.omni.npz'
#    omniave = opts.omnipath + 'omniave.npz'
    if not os.path.exists(omnixx): raise IOError('%s not found' %omnixx)
    if not os.path.exists(omniyy): raise IOError('%s not found' %omniyy)
#    if not os.path.exists(omniave): raise IOError('%s not found' %omniave)
    gx = mp2cal.io.quick_load_gains(omnixx)
    gy = mp2cal.io.quick_load_gains(omniyy)
#    gave = mp2cal.wyl.quick_load_gains(omniave)
    omnisol = {'x':{}, 'y':{}}
    for a in gx['x'].keys():
#        if gx['x'][a].ndim == 2: gx['x'][a] = np.mean(gx['x'][a], axis=0)
#        ind = np.where(gains['x'][a]!=0)[0]
#        gres = np.zeros_like(gx['x'][a])
#        gres[ind] = gx['x'][a][ind] - gave['x'][a][ind]
#        gr = np.fft.rfft(gres.real)
#        gi = np.fft.rfft(gres.imag)
#        gr[5:] *= 0
#        gi[5:] *= 0
#        gfit = np.complex64(np.fft.irfft(gr) + 1j*np.fft.irfft(gi))
        omnisol['x'][a] = np.mean(gx['x'][a],axis=0)#(gave['x'][a]+gfit)
    for a in gy['y'].keys():
#        if gy['y'][a].ndim == 2: gy['y'][a] = np.mean(gy['y'][a], axis=0)
#        ind = np.where(gains['y'][a]!=0)[0]
#        gres = np.zeros_like(gy['y'][a])
#        gres[ind] = gy['y'][a][ind] - gave['y'][a][ind]
#        gr = np.fft.rfft(gres.real)
#        gi = np.fft.rfft(gres.imag)
#        gr[5:] *= 0
#        gi[5:] *= 0
#        gfit = np.complex64(np.fft.irfft(gr) + 1j*np.fft.irfft(gi))
        omnisol['y'][a] = np.mean(gy['y'][a],axis=0)#(gave['y'][a]+gfit)
#    omnisol = mp2cal.wyl.degen_project_FO(omnisol,antpos)
if opts.subtract: suffix = suffix + 'S'
writepath = opts.outpath + 'data' + '_' + suffix + '/'
if not os.path.exists(writepath): os.makedirs(writepath)
newfile = writepath + obsid.split('/')[-1] + '.uvfits'
if os.path.exists(newfile): raise IOError('   %s exists.  Skipping...' % newfile)

# Load data
print "Loading: " + obsid + ".uvfits"
uv = uvd.UVData()
uv.read_uvfits(obsid+'.uvfits')

# Apply cal
print "Applying cal ..."
for pp in range(uv.Npols):
    p1,p2 = aipy.miriad.pol2str[uv.polarization_array[pp]]
    for ii in range(uv.Nbls):
        a1 = uv.ant_1_array[ii]
        a2 = uv.ant_2_array[ii]
        gi = gains[p1][a1]
        gj = gains[p2][a2]
        if np.any(np.isnan(gi)) or np.any(np.isnan(gj)):
            uv.flag_array[ii::uv.Nbls,:,:,:] = True
            continue
        fi = np.where(gains[p1][a1]!=0)[0]
        fj = np.where(gains[p2][a2]!=0)[0]
        uv.data_array[ii::uv.Nbls,0,fi,pp] /= gi[fi]
        uv.data_array[ii::uv.Nbls,0,fj,pp] /= gj[fj].conj()
        if opts.omniapp:
            if a1 in omnisol[p1].keys():
                uv.data_array[ii::uv.Nbls,0,fi,pp] /= omnisol[p1][a1][fi]
            if a2 in omnisol[p2].keys():
                uv.data_array[ii::uv.Nbls,0,fj,pp] /= omnisol[p2][a2][fj].conj()
# Subtracting the model
if opts.subtract:
    print "Subtracting model ..."
    modelxx = readsav(opts.fhdpath + 'cal_prerun/vis_data/' + obsid + '_vis_model_XX.sav')
    uv.data_array[:,0,:,0] -= modelxx['vis_model_ptr']
    del modelxx
    modelyy = readsav(opts.fhdpath + 'cal_prerun/vis_data/' + obsid + '_vis_model_YY.sav')
    uv.data_array[:,0,:,1] -= modelyy['vis_model_ptr']
    del modelyy

# Average data in frequency
print "Averaging in frequency channel ..."
data1 = np.ma.masked_array(uv.data_array[:,:,0::2,:], uv.flag_array[:,:,0::2,:])
data2 = np.ma.masked_array(uv.data_array[:,:,1::2,:], uv.flag_array[:,:,1::2,:])
data = np.ma.masked_array([data1,data2])
uv.data_array = None
del data1, data2
data = np.mean(data,axis=0)
sample = np.ma.masked_array(uv.nsample_array[:,:,0::2,:], uv.flag_array[:,:,0::2,:]) + \
        np.ma.masked_array(uv.nsample_array[:,:,1::2,:], uv.flag_array[:,:,1::2,:])
uv.nsample_array = None
uv.data_array = data.data
uv.flag_array = data.mask
del data
uv.nsample_array = sample.data
del sample
uv.freq_array = (uv.freq_array[:,0::2]+uv.freq_array[:,1::2])/2
uv.Nfreqs /= 2
uv.channel_width *= 2

print "writing ..."
uv.write_uvfits(newfile)
