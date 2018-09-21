import sys, optparse, glob, numpy as np, mp2cal
c_light=299792458.
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
    freqs = d['freqs']
    if opts.o:
        omni = np.load('calibration/red/'+str(o)+'.'+pol+'.omni.npz')
    day = o / 86164
    if not data.has_key(day): data[day] = {}
    for a in d.keys():
        if not a[0].isdigit(): continue
        g = d[a]
        if(np.any(np.isnan(g))): continue
        if opts.o and a>56:
            try: g *= omni[a][0]
            except: pass
        if not data[day].has_key(a): data[day][a] = []
        data[day][a].append(g)
fit = {}
nf = freqs.size
band = (freqs[-1]-freqs[0]) * nf / (nf - 1)
for day in data.keys():
    fit[day] = {}
    gbp = []
    for a in data[day].keys():
        x = np.copy(data[day][a])
        md = np.ma.masked_array(x, np.zeros(x.shape, dtype=bool))
        ind = np.where(x==0)
        md.mask[ind] = True
        md = np.mean(md,axis=0)
        data[day][a] = md
        gbp.append(np.abs(md)/np.mean(np.abs(md)))
        fqc = np.arange(md.size)
        ind = np.logical_not(md.mask)
        x = fqc[ind]
        y = np.unwrap(np.angle(md.data[ind]))
        z = np.polyfit(x,y,1)
        fit[day][a] = np.exp(1j*(z[0]*fqc+z[1]))
    gbp = np.ma.masked_array(gbp)
    gbp = np.mean(gbp, axis=0)
    for a in data[day].keys():
        ant = int(a[:-1])
        amp = np.mean(data[day][a]) / np.mean(gbp)
        phs = 1.
        if(mp2cal.pos.tile_info[ant]['cable']==150):
            reftime = 300. / (c_light * mp2cal.pos.tile_info[ant]['vf'])
            res = data[day][a] / fit[day][a]
            nu = np.sum(np.logical_not(res.mask))
            res = res.data * np.logical_not(res.mask)
            dmode = 0.05
            nmode = 50
            modes = np.linspace(-dmode*nmode, dmode*nmode, 2*nmode+1) + band * reftime
            res = np.resize(res, (2*nmode+1, nf))
            freq_mat = np.resize(np.arange(nf), (2*nmode+1, nf))
            t1 = np.sum(np.sin(2*np.pi/nf*modes*freq_mat.T).T*np.angle(res), axis=1)
            t2 = np.sum(np.cos(2*np.pi/nf*modes*freq_mat.T).T*np.angle(res), axis=1)
            i = np.argmax(t1**2+t2**2)
            mi = modes[i]
            phase_ripple = 2*t1[i]*np.sin(2*np.pi*(mi*np.arange(nf)/nf))/nu + \
                           2*t2[i]*np.cos(2*np.pi*(mi*np.arange(nf)/nf))/nu
            phs = np.exp(1j*phase_ripple)
        fit[day][a] *= amp*gbp.data*phs
        data[day][a] = data[day][a].data
    np.save('./calibration/gbp'+'F'+'O'*opts.o+'_'+str(day)+'_'+pol+'.npy', gbp.data)
    np.savez('./calibration/cal'+'F'+'O'*opts.o+'_'+str(day)+'_'+pol+'.npz' , **data[day])
    np.savez('./calibration/fit'+'F'+'O'*opts.o+'_'+str(day)+'_'+pol+'.npz' , **fit[day])

