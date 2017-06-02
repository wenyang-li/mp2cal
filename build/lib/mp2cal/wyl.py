import numpy as np, omnical, aipy
import subprocess, datetime, os
from astropy.io import fits
import copy
import heracal.omni as omni


def output_mask_array(flag_array):
    invf = 1 - flag_array
    sf = np.sum((np.sum(invf,axis=0)),axis=0).astype(bool)
    st = np.sum((np.sum(invf,axis=1)),axis=1).astype(bool)
    mask_array = 1 - np.outer(st,sf)
    mask_array = mask_array.astype(bool)
    return mask_array


def find_ex_ant(uvdata):
    ex_ant = []
    for ii in uvdata.antenna_numbers:
        if not ii in uvdata.ant_1_array and not ii in uvdata.ant_2_array:
            ex_ant.append(ii)
    return ex_ant


def uv_wrap_fc(uv,redbls,pols=['xx','yy']):
    wrap_list = []
    a1 = uv.ant_1_array[:uv.Nbls]
    a2 = uv.ant_2_array[:uv.Nbls]
    data = uv.data_array
    flag = uv.flag_array
    for jj in range(uv.Npols):
        pp = aipy.miriad.pol2str[uv.polarization_array[jj]]
        if not pp in pols: continue
        wrap = {}
        wrap['pol'] = pp
        wrap['data'] = {}
        wrap['flag'] = {}
        for ii in range(uv.Nbls):
            if (a1[ii],a2[ii]) in redbls: bl = (a1[ii],a2[ii])
            elif (a2[ii],a1[ii]) in redbls: bl = (a2[ii],a1[ii])
            else: continue
            if not wrap['data'].has_key(bl):
                if bl == (a1[ii],a2[ii]): dat_temp = data[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)[:,ii]
                else: dat_temp = data[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)[:,ii].conj()
                flg_temp = flag[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)[:,ii]
                dat_ma = np.ma.masked_array(dat_temp, mask=flg_temp)
                dat_ma = np.mean(dat_ma,axis=0)
                wrap['data'][bl] = np.complex64([dat_ma.data])
                wrap['flag'][bl] = np.array([dat_ma.mask])
        wrap_list.append(wrap)
    return wrap_list


def uv_wrap_omni(uv,pols=['xx','yy']):
    data_wrap = {}
    a1 = uv.ant_1_array[:uv.Nbls]
    a2 = uv.ant_2_array[:uv.Nbls]
    data = uv.data_array
    flag = uv.flag_array
    for jj in range(uv.Npols):
        pp = aipy.miriad.pol2str[uv.polarization_array[jj]]
        if not pp in pols: continue
        wrap = {}
        wrap['pol'] = pp
        wrap['data'] = {}
        wrap['flag'] = {}
        wrap['auto'] = {}
        wrap['mask'] = output_mask_array(flag[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs))
        auto_scale = 0
        for ii in range(uv.Nbls):
            if a1[ii] == a2[ii]:
                auto_m = np.ma.masked_array(data[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)[:,ii].real,mask=wrap['mask'])
                wrap['auto'][a1[ii]] = np.sqrt(np.mean(auto_m,axis=0).data) + 1e-10
                auto_scale += np.nanmean(wrap['auto'][a1[ii]])
            else:
                bl = (a1[ii],a2[ii])
                wrap['data'][bl][pp] = np.complex64(data[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)[:,ii])
                wrap['flag'][bl][pp] = np.array(flag[:,0][:,:,jj].reshape(uv.Ntimes,uv.Nbls,uv.Nfreqs)[:,ii])
        auto_scale /= len(wrap['auto'].keys())
        for a in wrap['auto'].keys(): wrap['auto'][a] /= auto_scale
        data_wrap[pp] = wrap
    return data_wrap


def polyfunc(x,z):
    sum = np.zeros((x.size))
    for ii in range(z.size):
        sum *= x
        sum += z[ii]
    return sum


def mwa_bandpass_fit(gains, auto, tile_info, amp_order=2, phs_order=1, fit_reflection=True):
    fqs = np.linspace(167.075,197.715,384)
    freq = np.arange(384)
    for p in gains.keys():
        for ant in gains[p].keys():
            x = np.where(gains[p][ant]!=0)[0]
            if x.size == 0: continue
            A = np.zeros((384),dtype=np.float)
            for n in range(0,24):
                chunk = np.arange(16*n+1,16*n+15)
                induse = np.where(gains[p][ant][chunk]!=0)
                z1 = np.polyfit(freq[chunk[induse]],np.abs(gains[p][ant][chunk[induse]])/auto[ant][chunk[induse]],amp_order)
                A[chunk[induse]] = auto[ant][chunk[induse]]*polyfunc(freq[chunk[induse]],z1)
            y2 = np.angle(gains[p][ant][x])
            y2 = np.unwrap(y2)
            z2 = np.polyfit(x,y2,phs_order)
            rp = np.zeros((384))
            cable = tile_info[ant]['cable']
            if fit_reflection and cable==150:
                vf = tile_info[ant]['vf']
                t0 = 2*cable/299792458.0/vf*1e6
                rp[x] = y2 - polyfunc(x,z2)
                tau = np.fft.fftfreq(384,(fqs[-1]-fqs[0])/383)
                fftrp = np.fft.fft(rp,n=384)
                inds = np.where(abs(np.abs(tau)-t0)<0.05)
                imax = np.argmax(np.abs(fftrp[inds]))
                ind = np.where(np.abs(tau)==np.abs(tau[inds][imax]))
                mask =np.zeros((384))
                mask[ind] = 1.
                fftrp *= mask
                rp = np.fft.ifft(fftrp)
            gains[p][ant][x] = A[x]*np.exp(1j*polyfunc(x,z2))
            gains[p][ant][x] *= np.exp(1j*rp[x])
    return gains


def poly_bandpass_fit(gains,amp_order=9, phs_order=1):
    for p in gains.keys():
        for a in gains[p].keys():
            SH = gains[p][a].shape
            g = copy.copy(gains[p][a])
#            g = np.mean(gains[p][a],axis=0)
            fqs = np.arange(g.size)
            fuse = np.where(g!=0)[0]
            z1 = np.polyfit(fuse,np.abs(g)[fuse],amp_order)
            z2 = np.polyfit(fuse,np.unwrap(np.angle(g)[fuse]),phs_order)
            gains[p][a] = polyfunc(fqs,z1)*np.exp(1j*polyfunc(fqs,z2))
            gains[p][a] = np.resize(gains[p][a],SH)
    return gains


def ampproj(g2,fhd):
    amppar = {}
    for p in g2.keys():
        s = 0
        n = 0
        SH = g2[p][g2[p].keys()[0]].shape
        for a in g2[p].keys():
            if np.isnan(np.mean(fhd[p][a])): continue
            s += (np.resize(np.abs(fhd[p][a]),SH)/np.abs(g2[p][a]))
            n += 1.
        amppar[p] = (s/n)
    return amppar


def phsproj(omni,fhd,realpos,EastHex,SouthHex,ref_antenna):
    phspar = {}
    ax1,ax2 = [],[]
    for ii in range(EastHex.shape[0]):
        ind_east = np.where(EastHex[ii]>0)[0]
        ind_south = np.where(SouthHex[ii]>0)[0]
        ax1.append(EastHex[ii][ind_east])
        ax1.append(SouthHex[ii][ind_south])
    for jj in range(EastHex.shape[1]):
        ind_east = np.where(EastHex[:,jj]>0)[0]
        ind_south = np.where(SouthHex[:,jj]>0)[0]
        ax2.append(EastHex[:,jj][ind_east])
        ax2.append(SouthHex[:,jj][ind_south])
    for p in omni.keys():
        phspar[p] = {}
        phspar[p]['phix'],phspar[p]['phiy'],phspar[p]['offset_east'],phspar[p]['offset_south'] = [],[],[],[]
        SH = omni[p][ref_antenna].shape
        for tt in range(0,SH[0]):
            slp1 = []
            slp2 = []
            for ff in range(0,SH[1]):
                if ff%16 in [0,15]:
                    slp1.append(0)
                    slp2.append(0)
                    continue
            #***** East-West direction fit *****#
                slope = []
                for inds in ax1:
                    x,tau = [],[]
                    for ii in inds:
                        if not ii in omni[p].keys(): continue
                        if np.isnan(fhd[p][ii][ff]): continue
                        x.append(realpos[ii]['top_x'])
                        tau.append(np.angle(fhd[p][ii][ff]/omni[p][ii][tt][ff]))
                    tau = np.unwrap(tau)
                    if tau.size < 3: continue
                    z = np.polyfit(x,tau,1)
                    slope.append(z[0])
                slope = np.array(slope)
                slp1.append(np.median(slope)) # slope could be steep, choosing median would be more likely to avoid phase wrapping
            #***** 60 deg East-South direction fit *****#
                slope = []
                for inds in ax2:
                    x,tau = [],[]
                    for ii in inds:
                        if not ii in omni[p].keys(): continue
                        if np.isnan(fhd[p][ii][ff]): continue
                        x.append(realpos[ii]['top_x'])
                        tau.append(np.angle(fhd[p][ii][ff]/omni[p][ii][tt][ff]))
                    tau = np.unwrap(tau)
                    if tau.size < 3: continue
                    z = np.polyfit(x,tau,1)
                    slope.append(z[0])
                slope = np.array(slope)
                slp2.append(np.median(slope))
        #****** calculate offset term ************#
            offset1, offset2 = [],[]
            phix = np.array(slp1)
            phiy = (np.array(slp2) - phix)/np.sqrt(3)
            for a in omni[p].keys():
                if np.isnan(np.mean(fhd[p][a])): continue
                dx = realpos[a]['top_x'] - realpos[ref_antenna]['top_x']
                dy = realpos[a]['top_y'] - realpos[ref_antenna]['top_y']
                proj = np.exp(1j*(dx*phix+dy*phiy))
                offset = np.exp(1j*np.angle(fhd[p][a]/omni[p][a][tt]/proj))
                if a < 93: offset1.append(offset)
                else: offset2.append(offset)
            offset1 = np.array(offset1)
            offset2 = np.array(offset2)
            offset1 = np.mean(offset1,axis=0)
            offset2 = np.mean(offset2,axis=0)
            phspar[p]['phix'].append(phix)
            phspar[p]['phiy'].append(phiy)
            phspar[p]['offset_east'].append(offset1)
            phspar[p]['offset_south'].append(offset2)
        phspar[p]['phix'] = np.array(phspar[p]['phix'])
        phspar[p]['phiy'] = np.array(phspar[p]['phiy'])
        phspar[p]['offset_east'] = np.array(phspar[p]['offset_east'])
        phspar[p]['offset_south'] = np.array(phspar[p]['offset_south'])
    return phspar


def cal_var_wgt(v,m,w):
    n = np.ma.masked_array(v-m,mask=w,fill_value=0.+0.j)
    var = np.var(n,axis=0).data
    zeros = np.where(var==0)
    var[zeros] = 1.
    inv = 1./var
    inv[zeros] = 0.
    return inv


def absoulte_cal(data,g2,realpos,ref_antenna,ex_ants=[],maxiter=50):
    gt = {}
    g3 = {}
    thred_length = 50*3e8/np.max(model_dict['freqs'])
    model_dict = data['model']
    for p in g2.keys():
        g3[p],gt[p] = {},{}
        a = g2[p].keys()[0]
        SH = g2[p][a].shape
        pp = p+p
        mvis = model_dict['data']
        mwgt = model_dict['flag']
        for a1 in range(0,57):
            nur,nui,den = 0,0,0
            if a1 in ex_ants: continue
            for a2 in g2[p].keys():
                sep = np.array([realpos[a2]['top_x']-realpos[a1]['top_x'],
                                realpos[a2]['top_y']-realpos[a1]['top_y'],
                                realpos[a2]['top_z']-realpos[a1]['top_z']])
                if np.linalg.norm(sep) < thred_length: continue
                bl = (a1,a2)
                try: dv = data[bl][pp]
                except(KeyError): dv = data[bl[::-1]][pp].conj()
                try:
                    dm = mvis[bl][pp]*(g2[p][a2].conj())
                    dw = np.logical_not(mwgt[bl][pp])
                except(KeyError):
                    dm = mvis[bl[::-1]][pp].conj()*(g2[p][a2].conj())
                    dw = np.logical_not(mwgt[bl[::-1]][pp])
                nur += np.nansum((dv.real*dm.real+dv.imag*dm.imag)*dw,axis=0)
                nui += np.nansum((dv.imag*dm.real-dv.real*dm.imag)*dw,axis=0)
                den += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dw,axis=0)
            if np.nansum(den) == 0: continue
            zeros = np.where(den==0)
            den[zeros] = 1.
            nur[zeros] = 0.
            nui[zeros] = 0.
            gt[p][a1] = nur/den + 1.j*nui/den
        for a2 in g2[p].keys():
            gt[p][a2] = copy.copy(g2[p][a2])
        for iter in range(maxiter):
            conv = 0
            #non-hex cal
            for a1 in gt[p].keys():
                if a1 > 56: continue
                nur,nui,den = 0,0,0
                for a2 in gt[p].keys():
                    if a2 < a1: continue
                    sep = np.array([realpos[a2]['top_x']-realpos[a1]['top_x'],
                                    realpos[a2]['top_y']-realpos[a1]['top_y'],
                                    realpos[a2]['top_z']-realpos[a1]['top_z']])
                    if np.linalg.norm(sep) < thred_length: continue
                    bl = (a1,a2)
                    try: dv = data[bl][pp]
                    except(KeyError): dv = data[bl[::-1]][pp].conj()
                    try:
                        dm = mvis[bl][pp]*(gt[p][a1]*gt[p][a2].conj())
                        dw = np.logical_not(mwgt[bl][pp])
                    except(KeyError):
                        dm = mvis[bl[::-1]][pp].conj()*(gt[p][a1]*gt[p][a2].conj())
                        dw = np.logical_not(mwgt[bl[::-1]][pp])
                    var_wgt = cal_var_wgt(dv,dm,np.logical_not(dw))
                    nur += np.nansum((dv.real*dm.real+dv.imag*dm.imag)*var_wgt*dw,axis=0)
                    nui += np.nansum((dv.imag*dm.real-dv.real*dm.imag)*var_wgt*dw,axis=0)
                    den += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*var_wgt*dw,axis=0)
                zeros = np.where(den==0)
                den[zeros] = 1.
                nur[zeros] = 0.
                nui[zeros] = 0.
                g3[p][a1] = (nur/den + 1.j*nui/den)*gt[p][a1]
    
            #degen parameter cal
            etan,etad = 0,0
            A,B,C,D,E,F,alpha,beta,gamma = 0,0,0,0,0,0,0,0,0
                    #*************************
                    #  [ A, B, D ] [phix]   [alpha]
                    #  [ B, C, E ] [phiy] = [beta ]
                    #  [ D, E, F ] [phi]    [gamma]
                    #*************************
            for a1 in gt[p].keys():
                if a1 < 57:
                    #non-hex_tile ---- hex_tile
                    for a2 in gt[p].keys():
                        if a2 < 57: continue
                        sep = np.array([realpos[a2]['top_x']-realpos[a1]['top_x'],
                                        realpos[a2]['top_y']-realpos[a1]['top_y'],
                                        realpos[a2]['top_z']-realpos[a1]['top_z']])
                        if np.linalg.norm(sep) < thred_length: continue
                        bl = (a1,a2)
                        try: dv = data[bl][pp]
                        except(KeyError): dv = data[bl[::-1]][pp].conj()
                        try:
                            dm = mvis[bl][pp]*(gt[p][a1]*gt[p][a2].conj())
                            dw = np.logical_not(mwgt[bl][pp])
                        except(KeyError):
                            dm = mvis[bl[::-1]][pp].conj()*(gt[p][a1]*gt[p][a2].conj())
                            dw = np.logical_not(mwgt[bl[::-1]][pp])
                        var_wgt = cal_var_wgt(dv,dm,np.logical_not(dw))
                        etan += np.nansum((dv.real*dm.real+dv.imag*dm.imag-dm.real*dm.real-dm.imag*dm.imag)*var_wgt*dw,axis=0)
                        etad += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*var_wgt*dw,axis=0)
                        dx = (realpos[a2]['top_x']-realpos[ref_antenna]['top_x'])/100.
                        dy = (realpos[a2]['top_y']-realpos[ref_antenna]['top_y'])/100.
                        A += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dx*dx*var_wgt*dw,axis=0)
                        B += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dx*dy*var_wgt*dw,axis=0)
                        C += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dy*dy*var_wgt*dw,axis=0)
                        alpha += np.nansum((dv.real*dm.imag-dv.imag*dm.real)*dx*var_wgt*dw,axis=0)
                        beta += np.nansum((dv.real*dm.imag-dv.imag*dm.real)*dy*var_wgt*dw,axis=0)
                        if a2 > 92:
                            D += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dx*var_wgt*dw,axis=0)
                            E += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dy*var_wgt*dw,axis=0)
                            F += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*var_wgt*dw,axis=0)
                            gamma += np.nansum((dv.real*dm.imag-dv.imag*dm.real)*var_wgt*dw,axis=0)
                else:
                    #hex_tile ---- hex_tile
                    for a2 in gt[p].keys():
                        if a2 < a1: continue
                        sep = np.array([realpos[a2]['top_x']-realpos[a1]['top_x'],
                                        realpos[a2]['top_y']-realpos[a1]['top_y'],
                                        realpos[a2]['top_z']-realpos[a1]['top_z']])
                        if np.linalg.norm(sep) < thred_length: continue
                        bl = (a1,a2)
                        try: dv = data[bl][pp]
                        except(KeyError): dv = data[bl[::-1]][pp].conj()
                        try:
                            dm = mvis[bl][pp]*(gt[p][a1]*gt[p][a2].conj())
                            dw = np.logical_not(mwgt[bl][pp])
                        except(KeyError):
                            dm = mvis[bl[::-1]][pp].conj()*(gt[p][a1]*gt[p][a2].conj())
                            dw = np.logical_not(mwgt[bl[::-1]][pp])
                        var_wgt = cal_var_wgt(dv,dm,np.logical_not(dw))
                        etan += np.nansum(2*(dv.real*dm.real+dv.imag*dm.imag-dm.real*dm.real-dm.imag*dm.imag)*var_wgt*dw,axis=0)
                        etad += np.nansum(4*(dm.real*dm.real+dm.imag*dm.imag)*var_wgt*dw,axis=0)
                        dx = (realpos[a1]['top_x']-realpos[a2]['top_x'])/100.
                        dy = (realpos[a1]['top_y']-realpos[a2]['top_y'])/100.
                        A += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dx*dx*var_wgt*dw,axis=0)
                        B += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dx*dy*var_wgt*dw,axis=0)
                        C += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dy*dy*var_wgt*dw,axis=0)
                        alpha -= np.nansum((dv.real*dm.imag-dv.imag*dm.real)*dx*var_wgt*dw,axis=0)
                        beta -= np.nansum((dv.real*dm.imag-dv.imag*dm.real)*dy*var_wgt*dw,axis=0)
                        if a1 < 93 and a2 > 92:
                            D -= np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dx*var_wgt*dw,axis=0)
                            E -= np.nansum((dm.real*dm.real+dm.imag*dm.imag)*dy*var_wgt*dw,axis=0)
                            F += np.nansum((dm.real*dm.real+dm.imag*dm.imag)*var_wgt*dw,axis=0)
                            gamma += np.nansum((dv.real*dm.imag-dv.imag*dm.real)*var_wgt*dw,axis=0)
            DET = A*C*F + 2*B*D*E - B*B*F - A*E*E - C*D*D
            zeros = np.where(DET == 0)
            DET[zeros] = 1
            phix = ((C*F-E*E)*alpha + (D*E-B*F)*beta + (B*E-C*D)*gamma)/DET
            phiy = ((D*E-B*F)*alpha + (A*F-D*D)*beta + (B*D-A*E)*gamma)/DET
            phi = ((B*E-C*D)*alpha + (B*D-A*E)*beta + (A*C-B*B)*gamma)/DET
            zeros = np.where(etad == 0)
            etad[zeros] = 1.
            etan[zeros] = 0.
            eta = etan/etad
            for a in g2[p].keys():
                dx = (realpos[a]['top_x']-realpos[ref_antenna]['top_x'])/100.
                dy = (realpos[a]['top_y']-realpos[ref_antenna]['top_y'])/100.
                if a < 93: projdegen = np.exp(eta+1j*(phix*dx+phiy*dy))
                else: projdegen = np.exp(eta+1j*(phix*dx+phiy*dy+phi))
                g3[p][a] = gt[p][a]*projdegen
            for a in gt[p].keys():
                conv += np.nanmean(np.abs(g3[p][a]-gt[p][a]))
            print 'check conv: ', iter, conv
            if conv < 0.05:
                print 'maxiter and conv for non-hex cal:',iter,conv
                break
            else:
                for a in gt[p].keys(): gt[p][a] = copy.copy(g3[p][a])
    return g3


def pos_to_info(position, pols=['x'], fcal=False, **kwargs):
    nant = position['nant']
    antpos = -np.ones((nant*len(pols),3))
    xmin,ymin = 0,0
    for key in position.keys():
        if key == 'nant': continue
        if position[key]['top_x'] < xmin: xmin = position[key]['top_x']
        if position[key]['top_y'] < ymin: ymin = position[key]['top_y']
    for ant in range(0,nant):
        try:
            x = position[ant]['top_x'] - xmin + 0.1
            y = position[ant]['top_y'] - ymin + 0.1
        except(KeyError): continue
        for z, pol in enumerate(pols):
            z = 2**z
            i = omni.Antpol(ant,pol,nant)
            antpos[i.val,0],antpos[i.val,1],antpos[i.val,2] = x,y,z
    reds = omni.compute_reds(nant, pols, antpos[:nant],tol=0.01)
    ex_ants = [Antpol(i,nant).ant() for i in range(antpos.shape[0]) if antpos[i,0] < 0]
    kwargs['ex_ants'] = kwargs.get('ex_ants',[]) + ex_ants
    reds = omni.filter_reds(reds, **kwargs)
    if fcal:
        info = omni.FirstCalRedundantInfo(nant)
    else:
        info = omni.RedundantInfo(nant)
    info.init_from_reds(reds, antpos)
    return info

def cal_reds_from_pos(position,**kwargs):
    nant = position['nant']
    antpos = -np.ones((nant,3))
    xmin = 0
    ymin = 0
    for key in position.keys():
        if key == 'nant': continue
        if position[key]['top_x'] < xmin: xmin = position[key]['top_x']
        if position[key]['top_y'] < ymin: ymin = position[key]['top_y']
    for ant in range(0,nant):
        try:
            x = position[ant]['top_x'] - xmin + 0.1
            y = position[ant]['top_y'] - ymin + 0.1
        except(KeyError): continue
        z = 0
        i = ant
        antpos[i,0],antpos[i,1],antpos[i,2] = x,y,z
    reds = omnical.arrayinfo.compute_reds(antpos,tol=0.01)
    kwargs['ex_ants'] = kwargs.get('ex_ants',[]) + [i for i in range(antpos.shape[0]) if antpos[i,0] < 0]
    reds = omnical.arrayinfo.filter_reds(reds,**kwargs)
    return reds

def get_phase(fqs,tau, offset=False):
    fqs = fqs.reshape(-1,1) #need the extra axis
    if offset:
        delay = tau[0]
        offset = tau[1]
        return np.exp(-1j*(2*np.pi*fqs*delay) - offset)
    else:
        return np.exp(-2j*np.pi*fqs*tau)

def save_gains_fc(s,fqs,outname):
    s2 = {}
    for k,i in s.iteritems():
        if len(i) > 1:
            s2[str(k)+pol] = get_phase(fqs,i,offset=True).T
            s2['d'+str(k)] = i[0]
            s2['o'+str(k)] = i[1]
        else:
            s2[str(k)+pol] = get_phase(fqs,i).T
            s2['d'+str(k)] = i
    np.savez(outname,**s2)

def load_gains_fc(fcfile):
    g0 = {}
    fc = np.load(fcfile)
    for k in fc.keys():
        if k[0].isdigit():
            a = int(k[:-1])
            p = k[-1]
            if not g0.has_key(p): g0[p] = {}
            g0[p][a] = fc[k]
    return g0

def save_gains_omni(filename, meta, gains, vismdl, xtalk):
    d = {}
    metakeys = ['jds','lsts','freqs','history']
    for key in meta:
        if key.startswith('chisq'): d[key] = meta[key] #separate if statements  pending changes to chisqs
        for k in metakeys:
            if key.startswith(k): d[key] = meta[key]
    for pol in gains:
        for ant in gains[pol]:
            d['%d%s' % (ant,pol)] = gains[pol][ant]
    for pol in vismdl:
        for bl in vismdl[pol]:
            d['<%d,%d> %s' % (bl[0],bl[1],pol)] = vismdl[pol][bl]
    for pol in xtalk:
        for bl in xtalk[pol]:
            d['(%d,%d) %s' % (bl[0],bl[1],pol)] = xtalk[pol][bl]
    np.savez(filename,**d)

def load_gains_omni(filename):
    meta, gains, vismdl, xtalk = {}, {}, {}, {}
    def parse_key(k):
        bl,pol = k.split()
        bl = tuple(map(int,bl[1:-1].split(',')))
        return pol,bl
    npz = np.load(filename)
    for k in npz.files:
        if k[0].isdigit():
            pol,ant = k[-1:],int(k[:-1])
            if not gains.has_key(pol): gains[pol] = {}
            gains[pol][ant] = npz[k]
        try: pol,bl = parse_key(k)
        except(ValueError): continue
        if k.startswith('<'):
            if not vismdl.has_key(pol): vismdl[pol] = {}
            vismdl[pol][bl] = npz[k]
        elif k.startswith('('):
            if not xtalk.has_key(pol): xtalk[pol] = {}
            xtalk[pol][bl] = npz[k]
        kws = ['chi','hist','j','l','f']
        for kw in kws:
            for k in [f for f in npz.files if f.startswith(kw)]: meta[k] = npz[k]
    return meta, gains, vismdl, xtalk