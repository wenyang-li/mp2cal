#! /usr/bin/env python
import numpy as np, time, mp2cal, optparse, os, sys, warnings
from multiprocessing import Pool

o = optparse.OptionParser()
o.set_usage('omni_run_multi.py [options] obsid') #only takes 1 obsid
o.set_description(__doc__)
o.add_option('-p', dest='pol', default='xx,yy', help='Choose polarization (xx, yy, xy, yx) to include, separated by commas.')
o.add_option('--ubls', dest='ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ubls', dest='ex_ubls', default='', help='Unique baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--bls', dest='bls', default='', help='Baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_bls', dest='ex_bls', default='', help='Baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--ants', dest='ants', default='', help='Antennas to use, separated by commas (ex: 1,4,64,49).')
o.add_option('--ex_ants', dest='ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--filepath',dest='filepath',default='/users/wl42/data/wl42/RAWOBS/',type='string', help='Path to input uvfits files. Include final / in path.')
o.add_option('--omnipath',dest='omnipath',default='calibration/red/',type='string', help='Path to save solution files. Include final / in path.')
o.add_option('--fhdpath', dest='fhdpath', default='', type='string', help='path to fhd dir')
o.add_option('--metafits', dest='metafits', default='', type='string', help='path to metafits files')
o.add_option('--tave', dest='tave', default=False, action='store_true', help='Toggle: average data in time')
o.add_option('--conv', dest='conv', default=False, action='store_true', help='do fine iterations for further convergence')
o.add_option('--ex_dipole', dest='ex_dipole', default=False, action='store_true', help='Toggle: exclude tiles which have dead dipoles')
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
if not opts.ex_ants == '':
    ex_ants = ant_parse(opts.ex_ants)

if not len(args) == 1: raise IOError("Do not support multiple files.")
obsid = args[0]
print "OBSID: " + obsid

#********************************* path for output plots ******************************************
plot_path = opts.omnipath + 'plots/'
if not os.path.exists(plot_path):
    try: os.makedirs(plot_path)
    except: pass
gfit_diff = {}

#********************************** load fhd ***************************************************
fhd_sol_path = opts.fhdpath+'calibration/'+obsid+'_cal.sav'
if os.path.exists(fhd_sol_path):
    print "fhd solutions exist: " + fhd_sol_path
    gfhd = mp2cal.io.load_gains_fhd(opts.fhdpath+'calibration/'+obsid+'_cal.sav', raw=True)
    for p in gfhd.keys():
        for a in gfhd[p].keys():
            if np.any(np.isnan(gfhd[p][a])):
                if not a in ex_ants: ex_ants.append(a)
else:
    raise IOError("No sky calibration found. Please do sky calibration first")

#********************************** load and wrap data ******************************************
uv = mp2cal.io.read(opts.filepath+obsid+'.uvfits')
t_jd = uv.time_array[::uv.Nbls]
t_lst = uv.lst_array[::uv.Nbls]
freqs = uv.freq_array[0]
Ntimes = uv.Ntimes
data_list = []
for pol in pols:
    RD = mp2cal.data.RedData(pol)
    RD.get_ex_ants(ex_ants)
    if opts.ex_dipole:
        metafits_path = opts.metafits + obsid + '.metafits'
        RD.get_dead_dipole(metafits_path)
    RD.read_data(uv, tave = opts.tave)
    RD.apply_fhd(gfhd)
    print "ex_ants for "+pol+":", RD.dead
    data_list.append(RD)

#******************************** omni run *******************************************************
def omnirun(RD):
    p = RD.pol[0]
    flagged_fqs = np.sum(np.logical_not(RD.mask),axis=0).astype(bool)
    flag_bls = []
    for bl in RD.flag.keys():
        wgt_data = np.logical_not(RD.flag[bl][RD.pol])
        wgt_data = np.sum(wgt_data,axis=0) + np.logical_not(flagged_fqs)
        ind = np.where(wgt_data==0)
        if ind[0].size > 0: flag_bls.append(bl)
    print 'flagged baselines: ', flag_bls
    omnisol = opts.omnipath + obsid + '.' + RD.pol + '.omni.npz'
    #*********************** red info ******************************************
    info = mp2cal.wyl.pos_to_info(pols=[p],ubls=ubls,ex_ubls=ex_ubls,bls=bls, \
                                  ex_bls=ex_bls+flag_bls,ants=ants,ex_ants=RD.dead)

    #*********************** generate g0 ***************************************
    print '     setting g0 as units'
    g0 = {p:{}}
    for a in info.subsetant: g0[p][a] = np.ones((1,freqs.size),dtype=np.complex64)

    #*********************** Calibrate ******************************************
    start_time = time.time()
    print '   Run omnical'
    m2,g2,v2 = mp2cal.wyl.run_omnical(RD.data,info,gains0=g0, maxiter=500, conv=1e-12)
    if opts.conv:
        print '   do fine conv'
        g2,v2 = mp2cal.wyl.fine_iter(g2,v2,RD.data,RD.mask,info,conv=1e-5,maxiter=500)
    end_time = time.time()
    caltime = (end_time - start_time)/60.
    print '   time expense: ', caltime

    #*********************** project degeneracy *********************************
    gsky = {p: gfhd[p]}
    RD.get_gains(g2, v2, gsky)
    RD.gains.get_auto(uv)
    RD.gains.degen_project_to_unit()

    #*********************** Do bandpass fitting ********************************
    RD.gains.bandpass_fitting(include_red = False)
    gfit_diff[p] = {}
    for a in RD.gains.red[p].keys():
        gfit_diff[p][a] = np.zeros_like(RD.gains.gfit[p][a])
        ind = np.where(RD.gains.gfit[p][a]!=0)[0]
        gfit_diff[p][a][ind] = 1./RD.gains.gfit[p][a][ind]
    RD.gains.bandpass_fitting(include_red = True)
    for a in RD.gains.red[p].keys():
        gfit_diff[p][a] *= RD.gains.gfit[p][a]

    #************************* metadata parameters ***************************************
    RD.recover_model_vis_waterfall(info)
    RD.cal_chi_square(info, m2)
    RD.plot_chisq_per_bl(plot_path, obsid)
    m2['history'] = 'OMNI_RUN: '+' '.join(sys.argv) + '\n'
    m2['jds'] = t_jd
    m2['lsts'] = t_lst
    m2['freqs'] = freqs

    #************************** Saving cal ************************************************
    print '     saving %s' % omnisol
    mp2cal.io.save_gains_omni(omnisol, m2, RD.gains.red, RD.gains.mdl) #save raw calibrations

par = Pool(2)
npzlist = par.map(omnirun, data_list)
par.close()

plot_sols(gfit_diff, freqs/1e6, plot_path, obsid)
