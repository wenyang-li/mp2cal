#!//anaconda/bin/python
import numpy as np, time
import hera_cal, aipy, mp2cal
import optparse, os, sys, glob
from astropy.io import fits
import pickle, copy
from multiprocessing import Pool
from scipy.io.idl import readsav
import pyuvdata.uvdata as uvd
#from IPython import embed

o = optparse.OptionParser()
o.set_usage('omni_run_multi.py [options] obsid') #only takes 1 obsid
o.set_description(__doc__)
aipy.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--ubls', dest='ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ubls', dest='ex_ubls', default='', help='Unique baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--bls', dest='bls', default='', help='Baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_bls', dest='ex_bls', default='', help='Baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--ants', dest='ants', default='', help='Antennas to use, separated by commas (ex: 1,4,64,49).')
o.add_option('--ex_ants', dest='ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--ftype', dest='ftype', default='', type='string', help='Type of the input file, uvfits or fhd save files')
o.add_option('--omnipath',dest='omnipath',default='',type='string', help='Path to load firstcal files and to save solution files. Include final / in path.')
o.add_option('--fhdpath', dest='fhdpath', default='/users/wl42/data/wl42/FHD_out/fhd_MWA_PhaseII_EoR0/', type='string', help='path to fhd dir for projecting degen parameters, or fhd output visibilities if ftype is fhd.')
o.add_option('--metafits', dest='metafits', default='/users/wl42/data/wl42/Nov2016EoR0/', type='string', help='path to metafits files')
o.add_option('--tave', dest='tave', default=False, action='store_true', help='choose to average data over time before calibration or not')
o.add_option('--conv', dest='conv', default=False, action='store_true', help='do fine iterations for further convergence')
opts,args = o.parse_args(sys.argv[1:])

#*****************************************************************************
def bl_parse(bl_args):
    bl_list = []
    for bl in bl_args.split(','):
        i,j = bl.split('_')
        bl_list.append((int(i),int(j)))
    return bl_list

def ant_parse(ant_args):
    ant_list = []
    for a in ant_args.split(','): ant_list.append(int(a))
    return ant_list

#************************************************************************************************
exec('from %s import *'% opts.cal) # Including antpos, realpos, EastHex, SouthHex
pols = opts.pol.split(',')
ubls, ex_ubls, bls, ex_bls, ants, ex_ants = None, [], None, [], None, []
if not opts.ubls == '':
    ubls = bl_parse(opts.ubls)
    print '     ubls: ', ubls
if not opts.ex_ubls == '':
    ex_ubls = bl_parse(opts.ex_ubls)
    print '     ex_ubls: ', ex_ubls
if not opts.bls == '':
    bls = bl_parse(opts.bls)
    print '     bls: ', bls
if not opts.ex_bls == '':
    ex_bls = bl_parse(opts.ex_bls)
    print '     ex_bls: ', ex_bls
if not opts.ants == '':
    ants = ant_parse(opts.ants)
    print '   ants: ', ants
if not opts.ex_ants == '': ex_ants = ant_parse(opts.ex_ants)

#********************************** load and wrap data ******************************************
if not len(args) == 1: raise IOError('Do not support multiple files.')
obsid = args[0]
uv = uvd.UVData()
if opts.ftype == 'uvfits':
    uv.read_uvfits(obsid+'.uvfits',run_check=False,run_check_acceptability=False)
elif opts.ftype == 'fhd':
    uv.read_fhd(glob.glob(opts.fhdpath+'/vis_data/'+obsid+'*')+glob.glob(opts.fhdpath+'/metadata/'+obsid+'*'),use_model=False,run_check=False,run_check_acceptability=False)
else: IOError('invalid filetype, it should be uvfits or fhd')
data_wrap = mp2cal.wyl.uv_wrap_omni(uv,pols=pols)
t_jd = uv.time_array[::uv.Nbls]
t_lst = uv.lst_array[::uv.Nbls]
freqs = uv.freq_array[0]
SH = (uv.Ntimes, uv.Nfreqs)


#*********************************** ex_ants *****************************************************
ex_ants_find = mp2cal.wyl.find_ex_ant(uv)
del uv
for a in ex_ants_find:
    if not a in ex_ants: ex_ants.append(a)
print '     ex_ants: ', ex_ants


data_list = []
for pp in pols: data_list.append(data_wrap[pp])

def omnirun(data_wrap):
    pp = data_wrap['pol']
    p = pp[0]
    data = data_wrap['data']
    flag = data_wrap['flag']
    auto = data_wrap['auto']
    mask_arr = data_wrap['mask']
    flagged_fqs = np.sum(np.logical_not(mask_arr),axis=0).astype(bool)
    flag_bls = []
    for bl in flag.keys():
        wgt_data = np.logical_not(flag[bl][pp])
        wgt_data = np.sum(wgt_data,axis=0) + np.logical_not(flagged_fqs)
        ind = np.where(wgt_data==0)
        if ind[0].size > 0: flag_bls.append(bl)
    print 'flagged baselines: ', flag_bls
    omnisol = opts.omnipath + obsid + '.' + pp + '.omni.npz'
    info = mp2cal.wyl.pos_to_info(antpos,pols=[p],fcal=False,ubls=ubls,ex_ubls=ex_ubls,bls=bls,ex_bls=ex_bls+flag_bls,ants=ants,ex_ants=ex_ants)
    reds = info.get_reds()
    redbls = [bl for red in reds for bl in red]

    #*********************** organize data *************************************
    dat,wgts,xtalk = {}, {}, {}
    for bl in data.keys():
        i,j = bl
        if not (i in info.subsetant and j in info.subsetant): continue
        if bl in ex_bls+flag_bls: continue
        if opts.tave:
            m = np.ma.masked_array(data[bl][pp],mask=flag[bl][pp])
            m = np.mean(m,axis=0)
            dat[bl] = {pp: np.complex64(m.data.reshape(1,-1))}
        else:
            dat[bl] = {pp: np.copy(data[bl][pp])}

    #*********************** generate g0 ***************************************
    g0 = {p: {}}
    if opts.tave:
        for a in info.subsetant: g0[p][a] = np.ones((1,freqs.size),dtype=np.complex64)
    else:
        for a in info.subsetant: g0[p][a] = np.ones(SH,dtype=np.complex64)

    #*********************** Calibrate ******************************************
    wgts[pp] = {} #weights dictionary by pol
    for bl in flag:
        i,j = bl
        wgts[pp][(j,i)] = wgts[pp][(i,j)] = np.logical_not(flag[bl][pp]).astype(np.int)
    start_time = time.time()
    print '   Run omnical'
#    m2,g2,v2 = mp2cal.wyl.run_omnical(dat,info,gains0=g0, maxiter=500, conv=1e-9)
    m2,g2,v2 = hera_cal.omni.run_omnical(dat,info,gains0=g0, maxiter=500, conv=1e-12)
    if opts.conv:
        print '   do fine conv'
        g2,v2 = mp2cal.wyl.fine_iter(g2,v2,dat,info,conv=1e-6,maxiter=500,mwa_mask=False)
    end_time = time.time()
    caltime = (end_time - start_time)/60.
    print '   time expense: ', caltime
    xtalk = hera_cal.omni.compute_xtalk(m2['res'], wgts) #xtalk is time-average of residual

    #************************ Average cal solutions ************************************
    if not opts.tave:
        print '   compute chi-square'
        chisq = 0
        for r in reds:
            for bl in r:
                if v2[pp].has_key(bl): yij = v2[pp][bl]
            for bl in r:
                try: md = np.ma.masked_array(data[bl][pp],mask=mask_arr)
                except(KeyError): md = np.ma.masked_array(data[bl[::-1]][pp].conj(),mask=mask_arr)
                i,j = bl
                chisq += (np.abs(md.data-g2[p][i]*g2[p][j].conj()*yij))**2/(np.var(md,axis=0).data+1e-7)
        DOF = (info.nBaseline - info.nAntenna - info.ublcount.size)
        m2['chisq2'] = chisq / float(DOF)
        chi = m2['chisq2']
        m2['flags'] = mask_arr
        chi_mask = np.zeros(chi.shape,dtype=bool)
        ind = np.where(chi>1.2)
        chi_mask[ind] = True
        or_mask = np.logical_or(chi_mask,mask_arr)
    for a in g2[p].keys():
        if opts.tave:
            stack_mask = np.sum(np.logical_not(mask_arr),axis=0).astype(bool)
            g2[p[0]][a] *= stack_mask
        else:
            g_temp = np.ma.masked_array(g2[p][a],or_mask,fill_value=0.0)
            g2[p[0]][a] = g_temp.data

    #*********************** project degeneracy *********************************
    g2 = mp2cal.wyl.degen_project_FO(g2,antpos,v2,mwa_mask=False)

    #************************* metadata parameters ***************************************
    m2['history'] = 'OMNI_RUN: '+' '.join(sys.argv) + '\n'
    m2['jds'] = t_jd
    m2['lsts'] = t_lst
    m2['freqs'] = freqs

    #************************** Saving cal ************************************************
    print '     saving %s' % omnisol
    mp2cal.wyl.save_gains_omni(omnisol,m2,g2,v2,xtalk)

par = Pool(2)
npzlist = par.map(omnirun,data_list)
par.close()