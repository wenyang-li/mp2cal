import numpy as np, pyuvdata.uvdata as uvd
import aipy, mp2cal, sys, os, optparse, gc

### Options ###
o = optparse.OptionParser()
o.set_usage('cal_apply_average.py [options] obs')
o.set_description(__doc__)
o.add_option('--fhdpath', dest='fhdpath', default='', type='string',
             help='path to fhd dir for fhd output visibilities if ftype is fhd. Include final / in path.')
o.add_option('--omnipath',dest='omnipath',default='',type='string', help='Path to load omnical solution files. Include final / in path.')
o.add_option('--omniapp',dest='omniapp',default=False,action='store_true',
             help='Toggle: apply omnical solutions to hex tiles. Default=False')
o.add_option('--outpath', dest='outpath', default='', type='string',
             help='path to output calibrated data. Include final / in path.')
opts,args = o.parse_args(sys.argv[1:])


obsid = args[0]

# Getting cal solutions
suffix = 'AF'
fhdsav = opts.fhdpath + 'calibration/' + obsid + '_cal.sav'
if not os.path.exists(fhdsav): raise IOError('%s not found' %fhdsav)
print "Get FHD solutions ..."
gains = mp2cal.wyl.load_gains_fhd(fhdsav)
if opts.omniapp:
    suffix = 'AOF'
    print "Get omnical solutions ..."
    omnixx = opts.omnipath + obsid + '.xx.omni.npz'
    omniyy = opts.omnipath + obsid + '.yy.omni.npz'
    if not os.path.exists(omnixx): raise IOError('%s not found' %omnixx)
    if not os.path.exists(omniyy): raise IOError('%s not found' %omniyy)
    gx = mp2cal.wyl.quick_load_gains(omnixx)
    gy = mp2cal.wyl.quick_load_gains(omniyy)
    for a in gx['x'].keys():
        if gx['x'][a].ndim == 2: gains['x'][a] = np.mean(gx['x'][a], axis=0)
        else: gains['x'][a] = gx['x'][a]
    for a in gy['y'].keys():
        if gy['y'][a].ndim == 2: gains['y'][a] = np.mean(gy['y'][a], axis=0)
        else: gains['y'][a] = gy['y'][a]
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

# Average data in frequency
print "Averaging in frequency channel ..."
data1 = np.ma.masked_array(uv.data_array[:,:,0::2,:], uv.flag_array[:,:,0::2,:])
data2 = np.ma.masked_array(uv.data_array[:,:,1::2,:], uv.flag_array[:,:,1::2,:])
data = np.ma.masked_array([data1,data2])
uv.data_array = None
del data1, data2
gc.collect()
data = np.mean(data,axis=0)
sample = np.ma.masked_array(uv.nsample_array[:,:,0::2,:], uv.flag_array[:,:,0::2,:]) + \
        np.ma.masked_array(uv.nsample_array[:,:,1::2,:], uv.flag_array[:,:,1::2,:])
uv.nsample_array = None
gc.collect()
uv.data_array = data.data
uv.flag_array = data.mask
del data
uv.nsample_array = sample.data
del sample
gc.collect()
uv.freq_array = (uv.freq_array[:,0::2]+uv.freq_array[:,1::2])/2
uv.Nfreqs /= 2
uv.channel_width *= 2

print "writing ..."
uv.write_uvfits(newfile)