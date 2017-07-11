import numpy as np
import sys, glob, mp2cal, optparse
from astropy.io import fits
from scipy.io.idl import readsav
delays = {
'0,5,10,15,1,6,11,16,2,7,12,17,3,8,13,18':-5,
'0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15':-4,
'0,3,6,9,0,3,6,9,0,3,6,9,0,3,6,9':-3,
'0,2,4,6,0,2,4,6,0,2,4,6,0,2,4,6':-2,
'0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3':-1,
'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0':0,
'3,2,1,0,3,2,1,0,3,2,1,0,3,2,1,0':1,
'6,4,2,0,6,4,2,0,6,4,2,0,6,4,2,0':2,
'9,6,3,0,9,6,3,0,9,6,3,0,9,6,3,0':3,
'12,8,4,0,13,9,5,1,14,10,6,2,15,11,7,3':4,
'15,10,5,0,16,11,6,1,17,12,7,2,18,13,8,3':5,
}

o = optparse.OptionParser()
o.set_usage('average.py [options]')
o.set_description(__doc__)
o.add_option('--fhd_path', dest='fhd_path', default='/users/wl42/data/wl42/FHD_out/fhd_PhaseII_Longrun_EoR0/calibration/', help='path to fhd cal solutions')
o.add_option('--dat_path', dest='fhd_path', default='/users/wl42/data/wl42/Nov2016EoR0/', help='path to fhd cal solutions')
opts,args = o.parse_args(sys.argv[1:])
#p = sys.argv[1]
pols = ['xx','yy']
meta, vismdl, xtalk = {},{},{}

fn=glob.glob(opts.fhd_path+'*_cal.sav')
fn.sort()
g = {}
#    nfiles = {}
for f in fn:
    cals = readsav(f,python_dict=True)
    obs = f.split('/')[-1].split('_')[0]
    metafits = opts.dat_path+obs+'.metafits'
    hdu = fits.open(metafits)
    day = int(obs)/86400
    suffix = str(day)+'_'+str(delays[hdu[0].header['DELAYS']])
    if not g.has_key(suffix): g[suffix]={'x':[],'y':[]}
    g[suffix]['x'].append(cals['cal']['GAIN'][0][0])
    g[suffix]['y'].append(cals['cal']['GAIN'][0][1])

for suf in g.keys():
    g[suffix]['x'] = np.array(g[suffix]['x'])
    g[suffix]['y'] = np.array(g[suffix]['y'])
    g[suffix]['x'] = np.nanmean(g[suffix]['x'],axis=0)
    g[suffix]['y'] = np.nanmean(g[suffix]['y'],axis=0)
    gx = {'x': {}}
    gy = {'y': {}}
    for a in range(128):
        gx['x'][a] = g[suffix]['x'][a]
        gy['y'][a] = g[suffix]['y'][a]
    outfnx = 'fhdave_'+str(suf)+'.xx.npz'
    outfny = 'fhdave_'+str(suf)+'.yy.npz'
    mp2cal.wyl.save_gains_omni(outfnx, meta, gx, vismdl, xtalk)
    mp2cal.wyl.save_gains_omni(outfny, meta, gy, vismdl, xtalk)

