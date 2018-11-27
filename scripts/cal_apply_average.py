from matplotlib import use
use('Agg')
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
suffix = 'AF'+'O'*opts.omniapp
if opts.subtract: suffix = suffix + 'S'
writepath = opts.outpath + 'data' + '_' + suffix + '/'
if not os.path.exists(writepath):
    try: os.makedirs(writepath)
    except: pass
newfile = writepath + obsid.split('/')[-1] + '.uvfits'
if os.path.exists(newfile): raise IOError('   %s exists.  Skipping...' % newfile)

# Load data
print "Loading: " + obsid + ".uvfits"
uv = uvd.UVData()
uv.read_uvfits(obsid+'.uvfits')
freqs = uv.freq_array[0]
graw = mp2cal.gain.RedGain(freqs = freqs)
gfhd = mp2cal.io.load_gains_fhd(opts.fhdpath+'calibration/'+obsid+'_cal.sav', raw=True)
graw.get_sky(gfhd)
if opts.omniapp:
    gomni = {'x': {}, 'y': {}}
    gx = mp2cal.io.quick_load_gains(opts.omnipath+obsid+'.xx.omni.npz')
    gy = mp2cal.io.quick_load_gains(opts.omnipath+obsid+'.yy.omni.npz')
    gomni['x'] = gx['x']
    gomni['y'] = gy['y']
    graw.get_red(gomni)
graw.bandpass_fitting(include_red = opts.omniapp)
gains = graw.gfit
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

# Noise spectrum flagging
ins = mp2cal.qltm.INS(uv)
ins.outliers_flagging()
ins.time_flagging()
ins.coherence_flagging()
ins.apply_flagging()
ins.saveplots(writepath, obsid.split('/')[-1])
ins.savearrs(writepath, obsid.split('/')[-1])

# Write out uvfits
print "writing ..."
ins.uv.write_uvfits(newfile)
