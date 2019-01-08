import numpy as np
import sys, glob, mp2cal, optparse
from astropy.io import fits

o = optparse.OptionParser()
o.set_usage('average_omni.py [options] obslist')
o.set_description(__doc__)
o.add_option('--omnipath',dest='omnipath',default='omni_sol/',type='string', help='Path to load firstcal files and to save solution files. Include final / in path.')
opts,args = o.parse_args(sys.argv[1:])

obsfn = args[0] #in the form of poln(pol number)point(pointing number)
pol_lookup = {'5': 'xx', '6': 'yy'}
pol = pol_lookup[obsfn[4]]
p = pol[0]
gains = {p: {}}
fn = open(obsfn, 'rb')
for line in fn:
    obs = line.strip()
    g = mp2cal.io.quick_load_gains(opts.omnipath+obs+'.'+pol+'.omni.npz')
    for a in g[p].keys():
        if not gains[p].has_key(a): gains[p][a] = []
        gains[p][a].append(g[p][a].flatten())
for a in gains[p].keys():
    x = np.array(gains[p][a])
    ind = np.where(x==0)
    md = np.ma.masked_array(x, np.zeros(x.shape, dtype=bool))
    md.mask[ind] = True
    md = np.mean(md, axis=0)
    gains[p][a] = md.data
mp2cal.io.save_gains_omni(opts.omnipath+obsfn+'.'+pol+'.omni.npz', {}, gains, {})
