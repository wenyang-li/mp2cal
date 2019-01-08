import numpy as np, sys
from scipy.io.idl import readsav
obs = sys.argv[1]
fhd = '/users/wl42/data/wl42/FHD_out/fhd_Calibration_PhaseII/calibration/'
d = readsav(fhd+obs+'_cal.sav')
c = d['cal']
gx = {}
gy = {}
freq = c['freq'][0]
gx['freqs'] = freq
gy['freqs'] = freq
for a in range(0,128): 
    gfx = np.array(c['GAIN'][0][0][a])
    gfy = np.array(c['GAIN'][0][1][a])
    if(np.any(np.isnan(gfx)) or np.any(np.isnan(gfy))): continue
    indx = np.where(gfx==0)
    indy = np.where(gfy==0)
    grx = gfx + np.array(c['GAIN_RESIDUAL'][0][0][a])
    gry = gfy + np.array(c['GAIN_RESIDUAL'][0][1][a])
    grx[indx] = 0
    gry[indy] = 0
    gx[str(a)+'x'] = grx
    gy[str(a)+'y'] = gry
np.savez('/users/wl42/data/wl42/OBS0/calibration/sky/'+obs+'.xx.fhd.npz', **gx)
np.savez('/users/wl42/data/wl42/OBS0/calibration/sky/'+obs+'.yy.fhd.npz', **gy)
