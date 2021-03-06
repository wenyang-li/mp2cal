import numpy as np, matplotlib.pyplot as plt
from scipy.io.idl import readsav
import pyuvdata.uvdata as uvd
from pos import *

def read(filename):
    """
    pyuvdata changed the conjugate convention for uvfits, flip things back here.
    """
    uv = uvd.UVData()
    uv.read_uvfits(filename)
    uv.data_array = np.conj(uv.data_array)
    uv.uvw_array = -uv.uvw_array
    return uv

def write(uv, outname):
    """
    pyuvdata changed the conjugate convention for uvfits, flip things back here.
    """
    uv.data_array = np.conj(uv.data_array)
    uv.uvw_array = -uv.uvw_array
    uv.write_uvfits(outname, write_lst=False)

def load_gains_fc(fcfile):
    """
    This function loads firstcal soltions created by mp2cal/scripts/firstcal_mwa.py
    The fcfile is in npz format.
    """
    g0 = {}
    fc = np.load(fcfile)
    for k in fc.keys():
        if k[0].isdigit():
            a = int(k[:-1])
            p = k[-1]
            if not g0.has_key(p): g0[p] = {}
            g0[p][a] = fc[k]
    return g0

def save_gains_fc(s,fqs,outname):
    """
    This function writes out firstcal solutions into npz.
    s: firstcal soltions
    fqs: frequency array, in Hz
    """
    def get_phase(fqs,tau, offset=False):
        fqs = fqs.reshape(-1,1) #need the extra axis
        if offset:
            delay = tau[0]
            offset = tau[1]
            return np.exp(-1j*(2*np.pi*fqs*delay - offset))
        else:
            return np.exp(-2j*np.pi*fqs*tau)
    s2 = {}
    for k,i in s.iteritems():
        if len(i) > 1:
            s2[str(k)] = get_phase(fqs,i,offset=True).T
            s2['d'+str(k)] = i[0]
            s2['o'+str(k)] = i[1]
        else:
            s2[str(k)] = get_phase(fqs,i).T
            s2['d'+str(k)] = i
    np.savez(outname,**s2)

def load_gains_omni(filename):
    """
    This function loads in information of omnical gains, including
    metadata, gains, model vis
    """
    meta, gains, vismdl = {}, {}, {}
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
        elif k.startswith('<'):
            pol,bl = parse_key(k)
            if not vismdl.has_key(pol): vismdl[pol] = {}
            vismdl[pol][bl] = npz[k]
        else:
            meta[k] = npz[k]
    return meta, gains, vismdl

def save_gains_omni(filename, meta, gains, vismdl):
    """
    This function saves omnical solutions, including metadata,
    gains, vis model
    """
    d = {}
    for key in meta:
        d[key] = meta[key] #separate if statements  pending changes to chisqs
    for pol in gains:
        for ant in gains[pol]:
            d['%d%s' % (ant,pol)] = gains[pol][ant]
    for pol in vismdl:
        for bl in vismdl[pol]:
            d['<%d,%d> %s' % (bl[0],bl[1],pol)] = vismdl[pol][bl]
    np.savez(filename,**d)

def quick_load_gains(filename):
    """
    This function only loads firstcal or omnical gains from an npz file.
    """
    d = np.load(filename)
    gains = {}
    for k in d.keys():
        if k[0].isdigit():
            p = k[-1]
            if not gains.has_key(p): gains[p] = {}
            a = int(k[:-1])
            gains[p][a] = d[k]
    return gains

def getpointing(obs):
    """
    Get pointing number for a specific obsid for filename purpose. This is valid for 2016 Phase II EoR data.
    Need to recheck the numbers if you would like to apply it to other data sets.
    """
    sid = int(obs)%86164.1
    if sid<=24214.35: return 'n5'
    elif 24214.35<sid<=26292.00: return 'n4'
    elif 26292.00<sid<=28392.95: return 'n3'
    elif 28392.95<sid<=30322.70: return 'n2'
    elif 30322.70<sid<=32141.35: return 'n1'
    elif 32141.35<sid<=33951.70: return '0'
    elif 33951.70<sid<=35810.80: return '1'
    else: return '2' # modify this line if there are pointings greater than 2

def load_bp_txt(fn, pol):
    """
    Load global bandpass calculated by FHD. They are in txt format.
    Output: a dictionary containing global bp for each pol.
    """
    dic = {'x': 1, 'y': 2}
    n = dic[pol]
    f = open(fn,'rb')
    bp = []
    for line in f:
        l = line.split('\t')
        bp.append(float(l[n].strip()))
    bp = np.array(bp)
    return bp

def load_gains_fhd(fhdsav,raw=False):
    """
    Load FHD calibration gains. They are in sav file format.
    Set raw = True if you want the raw gain amplitudes instead of polynomial fitted amp.
    The phase of the gains is always linear fitted as they are physical, i.e., the phases
    are always associated with an antenna delay. Flagged channels are set to be 0s
    """
    fhd_cal = readsav(fhdsav,python_dict=True)
    gfhd = {'x':{},'y':{}}
    for a in range(fhd_cal['cal']['N_TILE'][0]):
        gfhd['x'][a] = np.copy(fhd_cal['cal']['GAIN'][0][0][a])
        gfhd['y'][a] = np.copy(fhd_cal['cal']['GAIN'][0][1][a])
        if raw:
            rx = np.copy(fhd_cal['cal']['GAIN_RESIDUAL'][0][0][a])
            ry = np.copy(fhd_cal['cal']['GAIN_RESIDUAL'][0][1][a])
            rx[np.where(gfhd['x'][a]==0)] = 0
            ry[np.where(gfhd['y'][a]==0)] = 0
            gfhd['x'][a] += rx
            gfhd['y'][a] += ry
    return gfhd

def load_fhd_global_bandpass(fhdpath, obsid):
    """
    Load averaged global bandpass created by mp2cal/scripts/ave_global_bp.py
    if not exist, load global bp for this single obsid.
    """
    gp = {}
    pointing = getpointing(obsid)
    day = int(obsid) / 86164
    for p in ['x', 'y']:
        try:
            gp[p] = np.load(fhdpath+'calibration/'+str(day)+'_p'+pointing+'_'+p+p+'.npy')
            print("get averaged bandpass for pol " + p)
        except:
            gp[p] = load_bp_txt(fhdpath+'calibration/'+obsid+'_bandpass.txt', p)
            print("averaged bandpass not found for pol "+p+", using bp from the obs")
    return gp

def plot_sols(gains, freq, outdir, obsname):
    ampmax = 0
    ampmin = 1e10
    phsmax = -np.pi
    phsmin = np.pi
    SH = freq.shape
    sols = {}
    for p in gains.keys():
        sols[p] = {}
        for a in gains[p].keys():
            iuse = np.where(gains[p][a]!=0)[0]
            ampmax = max(ampmax, np.max(np.abs(gains[p][a][iuse])))
            ampmin = min(ampmin, np.min(np.abs(gains[p][a][iuse])))
            phsmax = max(phsmax, np.max(np.angle(gains[p][a][iuse])))
            phsmin = min(phsmin, np.min(np.angle(gains[p][a][iuse])))
            tile = tile_info[a]['Tile']
            sols[p][tile] = gains[p][a]
    # label of x axis
    amplim = ampmax-ampmin
    phslim = phsmax-phsmin
    ampmax += 0.02*amplim
    ampmin -= 0.02*amplim
    phsmax += 0.02*phslim
    phsmin -= 0.02*phslim
    lx=[freq[37*freq.size/384],freq[287*freq.size/384]]
    lax=np.int32(np.round(lx))
    # label of y axis for amplitude
    ly=[ampmin,(ampmax+ampmin)/2,ampmax]
    lay=['%.2f'%(ampmin),'%.2f'%((ampmax+ampmin)/2),'%.2f'%(ampmax)]
    fig=plt.figure(figsize=(10,10))
    plt.suptitle(obsname,y=0.99,size=15.0)
    for ii in range(0,12):
        for jj in range(0,6):
            ind=ii*6+jj
            p=fig.add_subplot(12,6,ind+1)
            try: p.scatter(freq[iuse],np.abs(sols['x'][ind+1001][iuse]),color='blue',s=0.005)
            except: pass
            try: p.scatter(freq[iuse],np.abs(sols['y'][ind+1001][iuse]),color='red',s=0.005)
            except: pass
            plt.ylim((ampmin,ampmax))
            plt.xlim((freq[0],freq[-1]))
            p.set_xticks(lx)
            p.set_yticks(ly)
            if jj==0: p.yaxis.set_ticklabels(lay,size=6.5)
            else: p.yaxis.set_ticklabels([])
            if ii==11: p.xaxis.set_ticklabels(lax,size=6.5)
            else: p.xaxis.set_ticklabels([])
            p.set_title(str(ind+1001),size=6.5,y=0.9)
    plt.subplots_adjust(top=0.93,bottom=0.05,left=0.035,right=0.98,hspace=0.55)
    fig.savefig(outdir+obsname+'_amp_omnical.png')

    # label of y axis for phase
    ly=[phsmin,(phsmax+phsmin)/2,phsmax]
    lay=['%.2f'%(phsmin),'%.2f'%((phsmax+phsmin)/2),'%.2f'%(phsmax)]
    fig=plt.figure(figsize=(10,10))
    plt.suptitle(obsname,y=0.99,size=15.0)
    for ii in range(0,12):
        for jj in range(0,6):
            ind=ii*6+jj
            p=fig.add_subplot(12,6,ind+1)
            try: p.scatter(freq[iuse],np.angle(sols['x'][ind+1001][iuse]),color='blue',s=0.005)
            except: pass
            try: p.scatter(freq[iuse],np.angle(sols['y'][ind+1001][iuse]),color='red',s=0.005)
            except: pass
            plt.ylim((phsmin,phsmax))
            plt.xlim((freq[0],freq[-1]))
            p.set_xticks(lx)
            p.set_yticks(ly)
            if jj==0: p.yaxis.set_ticklabels(lay,size=6.5)
            else: p.yaxis.set_ticklabels([])
            if ii==11: p.xaxis.set_ticklabels(lax,size=6.5)
            else: p.xaxis.set_ticklabels([])
            p.set_title(str(ind+1001),size=6.5,y=0.9)
    plt.subplots_adjust(top=0.93,bottom=0.05,left=0.035,right=0.98,hspace=0.55)
    fig.savefig(outdir+obsname+'_phs_omnical.png')
