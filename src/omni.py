import numpy as np
import omnical

POL_TYPES = 'xylrabne'
# XXX this can't support restarts or changing # pols between runs
POLNUM = {}  # factor to multiply ant index for internal ordering
NUMPOL = {}

# dict for converting to polarizations
jonesLookup = {
    -5: (-5, -5),
    -6: (-6, -6),
    -7: (-5, -6),
    -8: (-6, -5)
}


def add_pol(p):
    '''Add's pols to the global POLNUM and NUMPOL dictionaries; used for creating Antpol objects.'''
    global NUMPOL
    assert(p in POL_TYPES)
    POLNUM[p] = len(POLNUM)
    NUMPOL = dict(zip(POLNUM.values(), POLNUM.keys()))


class Antpol:
    '''Defines an Antpol object that encodes an antenna number and polarization value.'''

    def __init__(self, *args):
        '''
        Creates an Antpol object.
        Args:
            ant (int): antenna number
            pol (str): polarization string. e.g. 'x', or 'y'
            nant(int): total number of antennas.
        '''
        try:
            ant, pol, nant = args
            if pol not in POLNUM:
                add_pol(pol)
            self.val, self.nant = POLNUM[pol] * nant + ant, nant
        except(ValueError):
            self.val, self.nant = args

    def antpol(self):
        return self.val % self.nant, NUMPOL[self.val // self.nant]

    def ant(self):
        return self.antpol()[0]

    def pol(self):
        return self.antpol()[1]

    def __int__(self):
        return self.val

    def __hash__(self):
        return self.ant()

    def __str__(self):
        return ''.join(map(str, self.antpol()))

    def __eq__(self, v):
        return self.ant() == v

    def __repr__(self):
        return str(self)


# XXX filter_reds w/ pol support should probably be in omnical
def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None, crosspols=None, ex_crosspols=None):
    '''
    Filter redundancies to include/exclude the specified bls, antennas, unique bl groups and polarizations.
    Assumes reds indices are Antpol objects.
    Args:
        reds: list of lists of redundant baselines as antenna pair tuples. e.g. [[(1,2),(2,3)], [(1,3)]]
        bls (optional): list of baselines as antenna pair tuples to include in reds.
        ex_bls (optional): list of baselines as antenna pair tuples to exclude in reds.
        ants (optional): list of antenna numbers (as int's) to include in reds.
        ex_ants (optional): list of antenna numbers (as int's) to exclude in reds.
        ubls (optional): list of baselines representing their redundant group to include in reds.
        ex_ubls (optional): list of baselines representing their redundant group to exclude in reds.
        crosspols (optional): cross polarizations to include in reds. e.g. 'xy' or 'yx'.
        ex_crosspols (optional): cross polarizations to exclude in reds. e.g. 'xy' or 'yx'.
    Return:
        reds: list of lists of redundant baselines as antenna pair tuples.
    '''
    def pol(bl):
        return bl[0].pol() + bl[1].pol()
    if crosspols:
        reds = [r for r in reds if pol(r[0]) in crosspols]
    if ex_crosspols:
        reds = [r for r in reds if not pol(r[0]) in ex_crosspols]
    return omnical.arrayinfo.filter_reds(reds, bls=bls, ex_bls=ex_bls, ants=ants, ex_ants=ex_ants, ubls=ubls, ex_ubls=ex_ubls)


class RedundantInfo(omnical.calib.RedundantInfo):
    '''RedundantInfo object to interface with omnical. Includes support for Antpol objects.'''

    def __init__(self, nant, filename=None):
        '''Initialize with base clas __init__ and number of antennas.
        Args:
            nant (int): number of antennas.
            filename (str): filename (str) for legacy info objects.
        '''
        omnical.info.RedundantInfo.__init__(self, filename=filename)
        self.nant = nant

    def bl_order(self):
        '''Returns expected order of baselines.
        Return:
            (i,j) baseline tuples in the order that they should appear in data.
            Antenna indicies are in real-world order
            (as opposed to the internal ordering used in subsetant).
        '''
        return [(Antpol(self.subsetant[i], self.nant), Antpol(self.subsetant[j], self.nant)) for (i, j) in self.bl2d]

    def order_data(self, dd):
        """Create a data array ordered for use in _omnical.redcal.
        Args:
            dd (dict): dictionary whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
                       the conjugation convention of the provided data.  'dd' values are 2D arrays of (time,freq) data.
        Return:
            array: array whose ordering reflects the internal ordering of omnical. Used to pass into pack_calpar
        """
        d = []
        for i, j in self.bl_order():
            bl = (i.ant(), j.ant())
            pol = i.pol() + j.pol()
            try:
                d.append(dd[bl][pol])
            except(KeyError):
                d.append(dd[bl[::-1]][pol[::-1]].conj())
        return np.array(d).transpose((1, 2, 0))

    def pack_calpar(self, calpar, gains=None, vis=None, **kwargs):
        ''' Pack a calpar array for use in omnical.
        Note that this function includes polarization support by wrapping
        into calpar format.
        Args:
            calpar (array): array whose size is given by self.calpar_size. Usually initialized to zeros.
            gains (dict): dictionary of starting gains for omnical run. dict[pol][antenna]
            vis (dict): dictionary of starting visibilities (for a redundant group) for omnical run. dict[pols][bl],
            nondegenerategains dict(): gains that don't have a degeneracy component to them (e.g. firstcal gains).
                                       The gains get divided out before handing off calpar to omnical.
        Returns:
            calpar (array): The populated calpar array.
        '''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        if gains:
            _gains = {}
            for pol in gains:
                for i in gains[pol]:
                    ai = Antpol(i, pol, self.nant)
                    if nondegenerategains is not None:
                        # This conj is necessary to conform to omnical conj
                        # conv.
                        _gains[int(ai)] = gains[pol][i].conj() / nondegenerategains[pol][i].conj()
                    else:
                        # This conj is necessary to conform to omnical conj
                        # conv.
                        _gains[int(ai)] = gains[pol][i].conj()
        else:
            _gains = gains

        if vis:
            _vis = {}
            for pol in vis:
                for i, j in vis[pol]:
                    ai, aj = Antpol(i, pol[0], self.nant), Antpol(
                        j, pol[1], self.nant)
                    _vis[(int(ai), int(aj))] = vis[pol][(i, j)]
        else:
            _vis = vis

        calpar = omnical.calib.RedundantInfo.pack_calpar(
            self, calpar, gains=_gains, vis=_vis)

        return calpar

    def unpack_calpar(self, calpar, **kwargs):
        '''Unpack the solved for calibration parameters and repack to antpol format
        Args:
            calpar (array): calpar array output from omnical.
            nondegenerategains (dict, optional): The nondegenerategains that were divided out in pack_calpar.
                These are multiplied back into calpar here. gain dictionary format.
        Return:
            meta (dict): dictionary of meta information from omnical. e.g. chisq, iters, etc
            gains (dict): dictionary of gains solved for by omnical. gains[pol][ant]
            vis (dict): dictionary of model visibilities solved for by omnical. vis[pols][blpair]
    '''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        meta, gains, vis = omnical.calib.RedundantInfo.unpack_calpar(
            self, calpar, **kwargs)

        def mk_ap(a):
            return Antpol(a, self.nant)
        if 'res' in meta:
            for i, j in meta['res'].keys():
                api, apj = mk_ap(i), mk_ap(j)
                pol = api.pol() + apj.pol()
                bl = (api.ant(), apj.ant())
                if pol not in meta['res']:
                    meta['res'][pol] = {}
                meta['res'][pol][bl] = meta['res'].pop((i, j))
        # XXX make chisq a nested dict, with individual antpol keys?
        for k in [k for k in meta.keys() if k.startswith('chisq')]:
            try:
                ant = int(k.split('chisq')[1])
                meta['chisq' + str(mk_ap(ant))] = meta.pop(k)
            except(ValueError):
                pass
        for i in gains.keys():
            ap = mk_ap(i)
            if ap.pol() not in gains:
                gains[ap.pol()] = {}
            gains[ap.pol()][ap.ant()] = gains.pop(i).conj()
            if nondegenerategains:
                gains[ap.pol()][ap.ant()] *= nondegenerategains[ap.pol()][ap.ant()]
        for i, j in vis.keys():
            api, apj = mk_ap(i), mk_ap(j)
            pol = api.pol() + apj.pol()
            bl = (api.ant(), apj.ant())
            if pol not in vis:
                vis[pol] = {}
            vis[pol][bl] = vis.pop((i, j))
        return meta, gains, vis


def compute_reds(nant, pols, *args, **kwargs):
    '''Compute the redundancies given antenna_positions and wrap into Antpol format.
    Args:
        nant: number of antennas
        pols: polarization labels, e.g. pols=['x']
        *args: args to be passed to omnical.arrayinfo.compute_reds, specifically
               antpos: array of antenna positions in order of subsetant.
        **kwargs: extra keyword arguments
    Return:
        reds: list of list of baselines as antenna tuples
       '''
    _reds = omnical.arrayinfo.compute_reds(*args, **kwargs)
    reds = []
    for pi in pols:
        for pj in pols:
            reds += [[(Antpol(i, pi, nant), Antpol(j, pj, nant))
                      for i, j in gp] for gp in _reds]
    return reds

def compute_xtalk(res, wgts):
    '''Estimate xtalk as time-average of omnical residuals.
    Args:
        res: omnical residuals.
        wgts: dictionary of weights to use in xtalk generation.
    Returns:
        xtalk (dict): dictionary of visibilities.
    '''
    xtalk = {}
    for pol in res.keys():
        xtalk[pol] = {}
        for key in res[pol]:
            r, w = np.where(wgts[pol][key] > 0, res[pol][key], 0), wgts[
                pol][key].sum(axis=0)
            w = np.where(w == 0, 1, w)
            xtalk[pol][key] = (r.sum(axis=0) / w).astype(res[pol]
                                                         [key].dtype)  # avg over time
    return xtalk
