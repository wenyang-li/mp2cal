import sys, optparse, glob, numpy as np

o = optparse.OptionParser()
o.set_usage('average_cal.py [options] obslist')
o.set_description(__doc__)
o.add_option('-o', dest='o', default=False, action='store_true', help='Toggle: apply omnical')
opts,args = o.parse_args(sys.argv[1:])
fn = args[0]
poldict = {'5': 'xx', '6': 'yy'}
pol = poldict[fn[4]]
fo = open(fn, 'rb')
obs = []
for line in fo: obs.append(int(line.strip()))
obs = np.array(obs)
data = {}
for o in obs:
    d = np.load('calibration/sky/'+str(o)+'.'+pol+'.fhd.npz')
    if opts.o:
        omni = np.load('calibration/red/'+str(o)+'.'+pol+'.omni.npz')
    day = o / 86164
    if not data.has_key(day): data[day] = {}
    for a in d.keys():
        g = d[a]
        if(np.any(np.isnan(g))): continue
        if opts.o and a>56:
            try: g *= omni[a][0]
            except: pass
        if not data[day].has_key(a): data[day][a] = []
        data[day][a].append(g)
for day in data.keys():
    for a in data[day].keys():
        x = np.copy(data[day][a])
        md = np.ma.masked_array(x, np.zeros(x.shape, dtype=bool))
        ind = np.where(x==0)
        md.mask[ind] = True
        data[day][a] = np.mean(md,axis=0).data
    np.savez('./calibration/cal'+'F'+'O'*opts.o+'_'+str(day)+'_'+pol+'.npz' , **data[day])

