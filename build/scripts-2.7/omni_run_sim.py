#!//anaconda/bin/python
import numpy as np, time
import hera_cal, aipy, mp2cal
import optparse, os, sys
from multiprocessing import Pool
import pyuvdata.uvdata as uvd

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
o.add_option('--omnipath',dest='omnipath',default='',type='string', help='Path to load firstcal files and to save solution files. Include final / in path.')
o.add_option('--tave', dest='tave', default=False, action='store_true', help='Toggle: average data in time')
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

if not len(args) == 1: raise IOError('Do not support multiple files.')
obsid = args[0]
print "OBSID: " + obsid

#********************************** load and wrap data ******************************************
uv = uvd.UVData()
uv.read_uvfits(obsid+'.uvfits', run_check=False, run_check_acceptability=False)
t_jd = uv.time_array[::uv.Nbls]
t_lst = uv.lst_array[::uv.Nbls]
freqs = uv.freq_array[0]
Ntimes = uv.Ntimes
data_list = []
for pol in pols:
    RD = mp2cal.data.RedData(pol)
    RD.get_ex_ants(ex_ants)
    RD.read_data(uv, tave = opts.tave)
    print "ex_ants for "+pol+":", RD.dead
    data_list.append(RD)
del uv

#******************************** omni run *******************************************************
def omnirun(RD):
    p = RD.pol[0]
    omnisol = opts.omnipath + obsid + '.' + RD.pol + '.omni.npz'
    #*********************** red info ******************************************
    info = mp2cal.wyl.pos_to_info(pols=[p],fcal=False,ubls=ubls,ex_ubls=ex_ubls,bls=bls, \
                                  ex_bls=ex_bls,ants=ants,ex_ants=RD.dead)

    #*********************** generate g0 ***************************************
    print '     setting g0 as units'
    g0 = {p:{}}
    for a in info.subsetant: g0[p][a] = np.ones((1,freqs.size),dtype=np.complex64)

    #*********************** Calibrate ******************************************
    start_time = time.time()
    print '   Run omnical'
    m2,g2,v2 = hera_cal.omni.run_omnical(RD.data,info,gains0=g0, maxiter=500, conv=1e-12)
    if opts.conv:
        print '   do fine conv'
        g2,v2 = mp2cal.wyl.fine_iter(g2,v2,RD.data,RD.mask,info,conv=1e-6,maxiter=500)
    end_time = time.time()
    caltime = (end_time - start_time)/60.
    print '   time expense: ', caltime
    
    #*********************** project degeneracy *********************************
    RD.get_gains(g2, v2)
    RD.gains.degen_project_FO()

    #************************* metadata parameters ***************************************
    m2['history'] = 'OMNI_RUN: '+' '.join(sys.argv) + '\n'
    m2['jds'] = t_jd
    m2['lsts'] = t_lst
    m2['freqs'] = freqs
    m2['flags'] = RD.mask
    #************************** Saving cal ************************************************
    print '     saving %s' % omnisol
    mp2cal.io.save_gains_omni(omnisol, m2, RD.gains.red, RD.gains.mdl)

par = Pool(2)
npzlist = par.map(omnirun, data_list)
par.close()

