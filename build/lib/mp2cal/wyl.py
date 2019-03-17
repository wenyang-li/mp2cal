import numpy as np, omnical, omni
from pos import *

def pos_to_info(pols=['x'], **kwargs):
    nant = antpos['nant']
    _antpos = -np.ones((nant*len(pols),3))
    xmin,ymin = 0,0
    for key in antpos.keys():
        if key == 'nant': continue
        if antpos[key]['top_x'] < xmin: xmin = antpos[key]['top_x']
        if antpos[key]['top_y'] < ymin: ymin = antpos[key]['top_y']
    for ant in range(0,nant):
        try:
            x = antpos[ant]['top_x'] - xmin + 0.1
            y = antpos[ant]['top_y'] - ymin + 0.1
        except(KeyError): continue
        for z, pol in enumerate(pols):
            z = 2**z
            i = omni.Antpol(ant,pol,nant)
            _antpos[i.val,0],_antpos[i.val,1],_antpos[i.val,2] = x,y,z
    reds = omni.compute_reds(nant, pols, _antpos[:nant],tol=0.01)
    ex_ants = [omni.Antpol(i,nant).ant() for i in range(_antpos.shape[0]) if _antpos[i,0] < 0]
    kwargs['ex_ants'] = kwargs.get('ex_ants',[]) + ex_ants
    reds = omni.filter_reds(reds, **kwargs)
    info = omni.RedundantInfo(nant)
    info.init_from_reds(reds, _antpos)
    return info

def cal_reds_from_pos(**kwargs):
    nant = antpos['nant']
    _antpos = -np.ones((nant,3))
    xmin = 0
    ymin = 0
    for key in antpos.keys():
        if key == 'nant': continue
        if antpos[key]['top_x'] < xmin: xmin = antpos[key]['top_x']
        if antpos[key]['top_y'] < ymin: ymin = antpos[key]['top_y']
    for ant in range(0,nant):
        try:
            x = antpos[ant]['top_x'] - xmin + 0.1
            y = antpos[ant]['top_y'] - ymin + 0.1
        except(KeyError): continue
        z = 0
        i = ant
        _antpos[i,0],_antpos[i,1],_antpos[i,2] = x,y,z
    reds = omnical.arrayinfo.compute_reds(_antpos,tol=0.01)
    kwargs['ex_ants'] = kwargs.get('ex_ants',[]) + [i for i in range(_antpos.shape[0]) if _antpos[i,0] < 0]
    reds = omnical.arrayinfo.filter_reds(reds,**kwargs)
    return reds

def GPR_interp(x,y,xp,col=1.4e5):
    # x has to be frequency array in Hz. col is coherence length in Hz, default=140kHz
    kk = np.resize(x,(x.size,x.size))
    pp = np.resize(xp,(xp.size,xp.size))
    kp = np.resize(x,(xp.size,x.size))
    pk = np.resize(xp,(x.size,xp.size))
    Kxx = np.exp(-((kk-kk.T)/col)**2)
    Kxp = np.exp(-((kp-pk.T)/col)**2)
    Kpx = Kxp.T
    Kpp = np.exp(-((pp-pp.T)/col)**2)
    return Kxp.dot(np.linalg.pinv(Kxx)).dot(y), Kpp - Kxp.dot(np.linalg.pinv(Kxx)).dot(Kpx)

def fit_data(data,flag,fit_order=2):
    md = np.ma.masked_array(data, flag)
    if data.ndim == 2:
        d = np.mean(md,axis=0)
    else: d = data
    x = np.arange(d.size)
    ind = np.where(d.mask==False)
    fr, sr = GaussianKernel(x[ind], d.data.real[ind], x)
    fi, si = GaussianKernel(x[ind], d.data.imag[ind], x)
    return fr + 1j*fi

def getfilter(filter_width = 3):
    w = filter_width
    Filter = np.zeros((2*w+1))
    for ii in range(2*w+1):
        Filter[ii][jj] = np.exp(-(ii-w)**2)
    return Filter

def impute_arr(vis, flg, mask_all, Filter, filter_width = 3):
    if np.sum(np.logical_not(flg))==0: return flg   #If the whole baseline is flagged, then do nothing
    w = filter_width
    md = np.ma.masked_array(vis, flg)
    wgt = np.logical_not(flg)
    ind = np.where(np.logical_xor(mask_all, flg))
    sz = ind[0].size
    if sz==0: return mask_all
    nt, nf = vis.shape
    flg2 = np.copy(mask_all)
    def interp(n):
        t = ind[0][n]
        f = ind[1][n]
        sf = max(0, f-w)
        ef = min(nf-1, f+w)
        fili = Filter[max(w-f,0):min(f+w,nf-1)-f+w+1]
        w0 = np.logical_not(flg[sf:ef+1])
        if np.sum(w0)==0:
            flg2[t, :] = True
        else:
            wgts = fili*w0
            vis[t][f] = np.sum(vis[sf:ef+1]*wgts) / np.sum(wgts)
    map(interp, np.arange(sz))
    return flg2

def impute_mwa(uv, filter_width = 3):
    Filter = getfilter(filter_width = filter_width)
    flags = uv.flag_array.reshape(uv.Ntimes, uv.Nbls, uv.Nfreqs, uv.Npols)
    for ii in range(uv.Npols):
        flg = flags[:,:,:,ii]
        t_slots = np.where(np.sum(np.logical_not(flg), axis=(1,2))==0)[0]
        f_slots = np.where(np.sum(np.logical_not(flg), axis=(0,1))==0)[0]
        mask_all = np.zeros((uv.Ntimes, uv.Nfreqs), dtype=bool)
        mask_all[t_slots,:] = True
        mask_all[:,f_slots] = True
        def impute_data(b):
            if uv.ant_1_array[b] == uv.ant_2_array[b]: return # Do nothing with autos
            vis = uv.data_array[b::uv.Nbls,0,:,ii]
            flg2 = impute_arr(vis, flg[:,b,:], mask_all, Filter, filter_width = filter_width)
            uv.data_array[b::uv.Nbls,0,:,ii] = vis
            uv.flag_array[b::uv.Nbls,0,:,ii] = flg2
        map(impute_data, np.arange(uv.Nbls))

def rough_cal(data,flag,info,pol='xx'): #The data has to be the averaged over time axis
    p = pol[0]
    g0 = {p: {}}
    phi = {}
    reds = info.get_reds()
    reds[0].sort()
    reds[1].sort()
    redbls = reds[0] + reds[1]
    redbls.sort()
    SH = data[reds[0][0]][pol].shape
    gamma0 = fit_data(data[reds[0][0]][pol], flag[reds[0][0]][pol])
    gamma1 = fit_data(data[reds[1][0]][pol], flag[reds[1][0]][pol])
    subsetant = info.subsetant
    fixants = (min(subsetant), min(subsetant[np.where(subsetant>92)]))
    for a in fixants: phi[a] = np.zeros(SH)
    while len(redbls) > 0:
        i,j = redbls[0]
        r = (i,j)
        redbls.remove(r)
        if phi.has_key(i) and phi.has_key(j): continue
        elif phi.has_key(i) and not phi.has_key(j):
            if r in reds[0]:
                phi[j] = np.angle(fit_data(data[r][pol],flag[reds[0][0]][pol])*np.exp(1j*phi[i])*gamma0.conj())
            elif r in reds[1]:
                phi[j] = np.angle(fit_data(data[r][pol],flag[reds[0][0]][pol])*np.exp(1j*phi[i])*gamma1.conj())
        elif phi.has_key(j) and not phi.has_key(i):
            if r in reds[0]:
                phi[i] = np.angle(fit_data(data[r][pol],flag[reds[0][0]][pol]).conj()*np.exp(1j*phi[j])*gamma0)
            elif r in reds[1]:
                phi[i] = np.angle(fit_data(data[r][pol],flag[reds[0][0]][pol]).conj()*np.exp(1j*phi[j])*gamma1)
        else: redbls.append(r)
    if len(phi.keys()) != subsetant.size: raise IOError('Missing antennas')
    for a in phi.keys():
        g0[p][a] = np.exp(-1j*phi[a])
    return g0


def run_omnical(data, info, gains0=None, xtalk=None, maxiter=500, conv=1e-3,
                     stepsize=.3, trust_period=1):
    
    m1,g1,v1 = omnical.calib.logcal(data, info, xtalk=xtalk, gains=gains0,
                                    maxiter=maxiter, conv=conv, stepsize=stepsize,
                                    trust_period=trust_period)
    m2,g2,v2 = omnical.calib.lincal(data, info, xtalk=xtalk, gains=g1, vis=v1,
                                    maxiter=maxiter, conv=conv, stepsize=stepsize,
                                    trust_period=trust_period)

    return m2,g2,v2


def unpack_linsolve(s):
    m, g, v = {}, {}, {}
    for key in s[0].keys(): m[key] = s[key]
    for key in s[1].keys():
        if len(key) == 2:
            a, p = key
            if not g.has_key(p): g[p] = {}
            g[p][a] = s[key]
        elif len(key) == 3:
            i, j, pp = key
            if not v.has_key(pp): v[pp] = {}
            v[pp][(i,j)] = s[key]
    return m, g, v

def fine_iter(g2,v2,data,mask,info,conv=1e-7,maxiter=500):
    for p in g2.keys():
        pp = p+p
        bl2d = []
        for ii in range(info.bl2d.shape[0]):
            bl2d.append(tuple(info.bl2d[ii]))
        a0 = g2[p].keys()[0]
        SH = g2[p][a0].shape
        mask_arr = mask.flatten()
        gs = {}
        vs = {}
        ant_map = {}
        ubl_map = {}
        for a in g2[p].keys():
            ai = info.ant_index(a)
            ant_map[ai] = a
            gs[ai] = g2[p][a].flatten()
        for bl in v2[pp].keys():
            i0 = info.ant_index(bl[0])
            j0 = info.ant_index(bl[1])
            bli = bl2d.index((i0,j0))
            ubli = info.bltoubl[bli]
            vs[ubli] = v2[pp][bl].flatten()
            ubl_map[bl] = ubli
        def grad_decent(ii):
            if mask_arr[ii]: return #specific for mwa
            dt = ii/SH[1]
            df = ii%SH[1]
            nbls = len(bl2d)
            na = info.nAntenna
            nubl = len(info.ublcount)
            A = np.zeros((2*nbls,2*(na+nubl)),dtype=np.float32)
            M = np.zeros((2*nbls),dtype=np.float32)
            S = np.zeros((2*(na+nubl)),dtype=np.float32)
            componentchange = 100
            def buildM(b):
                a1,a2 = bl2d[b]
                u = info.bltoubl[b]
                gj_yij = gs[a2][ii].conj()*vs[u][ii]
                giyij = gs[a1][ii]*vs[u][ii]
                gigj_ = gs[a1][ii]*gs[a2][ii].conj()
                try: dvij = data[(ant_map[a1],ant_map[a2])][pp][dt][df] - gigj_*vs[u][ii]
                except(KeyError): dvij = data[(ant_map[a2],ant_map[a1])][pp][dt][df].conj() - gigj_*vs[u][ii]
                A[2*b,2*a1] = gj_yij.real
                A[2*b,2*a1+1] = -gj_yij.imag
                A[2*b+1,2*a1] = gj_yij.imag
                A[2*b+1,2*a1+1] = gj_yij.real
                A[2*b,2*a2] = giyij.real
                A[2*b,2*a2+1] = giyij.imag
                A[2*b+1,2*a2] = giyij.imag
                A[2*b+1,2*a2+1] = -giyij.real
                A[2*b,2*na+2*u] = gigj_.real
                A[2*b,2*na+2*u+1] = -gigj_.imag
                A[2*b+1,2*na+2*u] = gigj_.imag
                A[2*b+1,2*na+2*u+1] = gigj_.real
                M[2*b] = dvij.real
                M[2*b+1] = dvij.imag
            def updata_sol(n):
                ds = np.complex64(S[2*n] + 1j*S[2*n+1])
                if n < na:
                    gs[n][ii] += ds
                    fchange = np.abs(ds/gs[n][ii])
                else:
                    vs[n-na][ii] += ds
                    fchange = np.abs(ds/vs[n-na][ii])
                return fchange
            for iter3 in range(maxiter):
                map(buildM,np.arange(nbls))
                S = np.linalg.pinv(A.transpose().dot(A),rcond=1e-8).dot(A.transpose()).dot(M)
                componentchange = np.max(map(updata_sol,np.arange(na+nubl)))
                if componentchange < conv: break
            print (dt,df),"  fine iter: ", iter3, "  conv: ", componentchange
        map(grad_decent, range(SH[0]*SH[1]))
        for a in g2[p].keys():
            g2[p][a] = np.resize(gs[info.ant_index(a)],SH)
        for bl in v2[pp].keys():
            v2[pp][bl] = np.resize(vs[ubl_map[bl]],SH)
    return g2,v2
