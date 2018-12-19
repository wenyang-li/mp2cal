# edited version of firstcal in hera_cal
#! /usr/bin/env python
from matplotlib import use
use('Agg')
import hera_cal, mp2cal
import pylab as p, aipy as a
import sys, optparse, os
import numpy as np
from multiprocessing import Pool

o = optparse.OptionParser()
o.set_usage('firstcal_mwa.py [options] obsid')
a.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--filepath',dest='filepath',default='/users/wl42/data/wl42/RAWOBS/',type='string', help='Path to input uvfits files. Include final / in path.')
o.add_option('--ubls', dest='ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ubls', dest='ex_ubls', default='', help='Unique baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--bls', dest='bls', default='', help='Baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_bls', dest='ex_bls', default='', help='Baselines to exclude, separated by commas (ex: 1_4,64_49).')
o.add_option('--ants', dest='ants', default='', help='Antennas to use, separated by commas (ex: 1,4,64,49).')
o.add_option('--ex_ants', dest='ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--outpath', default='/users/wl42/data/wl42/Nov2016EoR0/omni_sol/',help='Output path of solutions.')
o.add_option('--verbose', action='store_true', default=False, help='Turn on verbose.')
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
if not opts.ex_ants == '': ex_ants = ant_parse(opts.ex_ants)

#********************************** load and wrap data ******************************************
if not len(args) == 1: raise IOError('Do not support multiple files.')
obsid = args[0]
uv = mp2cal.io.read(opts.filepath+obsid+'.uvfits')
data_list = []
for pol in pols:
    RD = mp2cal.data.RedData(pol)
    RD.get_ex_ants(ex_ants)
    RD.read_data(uv, tave = True)
    print "ex_ants for "+pol+":", RD.dead
    data_list.append(RD)
fqs = uv.freq_array[0]/1e9
del uv

#************************************ run firstcal ************************************************
def firstcal(RD):
    flagged_fqs = np.sum(np.logical_not(RD.mask),axis=0).astype(bool)
    flag_bls = []
    for bl in RD.flag.keys():
        wgt_data = np.logical_not(RD.flag[bl][RD.pol])
        wgt_data = np.sum(wgt_data,axis=0) + np.logical_not(flagged_fqs)
        ind = np.where(wgt_data==0)
        if ind[0].size > 0: flag_bls.append(bl)
    print 'flagged baselines: ', flag_bls
    outname = opts.outpath + obsid + '.' + RD.pol + '.fc.npz'
    #******************************** red info ***************************************************
    info = mp2cal.wyl.pos_to_info(pols=[RD.pol[0]],fcal=True,ubls=ubls,ex_ubls=ex_ubls,bls=bls, \
                                  ex_bls=ex_bls+flag_bls,ants=ants,ex_ants=RD.dead)
    wgtpack = {k : { qp : np.logical_not(RD.flag[k][qp]) for qp in RD.flag[k]} for k in RD.flag}
    fc = hera_cal.firstcal.FirstCal(RD.data,wgtpack,fqs,info)
    print "     running firstcal"
    sols = fc.run(finetune=True,verbose=False,average=True,window='none')
    print('     Saving {0}'.format(outname))
    mp2cal.io.save_gains_fc(sols,fqs*1e9,outname)

#*****************************************************************************
par = Pool(2)
npzlist = par.map(firstcal, data_list)
par.close()
