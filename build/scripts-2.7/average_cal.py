import sys, mp2cal, numpy as np
inpath = './' # I will make it as an option later
outpath = './'
fn = open(sys.argv[1], 'rb')
g = {'x': {}, 'y':{}}
fq = None
for line in fn:
    obs = line.strip()
    gx = mp2cal.io.quick_load_gains(inpath + obs + '.xx.omni.npz')
    gy = mp2cal.io.quick_load_gains(inpath + obs + '.yy.omni.npz')
    if fq is None:
        d0 = np.load(inpath + obs + '.xx.omni.npz')
        fq = d0['freqs']
    for a in gx['x'].keys():
        if g['x'].has_key(a):
            g['x'][a].append(gx['x'][a])
        else:
            g['x'][a] = [gx['x'][a]]
    for a in gy['y'].keys():
        if g['y'].has_key(a):
            g['y'][a].append(gy['y'][a])
        else:
            g['y'][a] = [gy['y'][a]]
for p in g.keys():
    for a in g[p].keys():
        g[p][a] = np.array(g[p][a])
        ind = np.logical_or(g[p][a]==1, g[p][a]==0)
        d = np.ma.masked_array(g[p][a], np.zeros(g[p][a], dtype=bool))
        d.mask[ind] = True
        g[p][a] = np.mean(d,axis=0).data
mp2cal.io.save_gains_omni(outpath + sys.argv[1].split('/')[-1] + '_ave.npz',{},g,{})
mp2cal.io.plot_sols(g, fq, outpath, sys.argv[1].split('/')[-1])
