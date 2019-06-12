import numpy as np

def cal_dim(reds):
    nbls = 0
    a2n = {}
    ants = []
    for r in reds:
        n = len(r)
        nbls += n*(n-1)/2
        for b in r:
            i,j = b
            if not i in ants: ants.append(i)
            if not j in ants: ants.append(j)
    ants.sort()
    for ii in range(len(ants)):
        a2n[ants[ii]] = ii
    return nbls, ants, a2n

def adaptfit(x0, y0):
    x = np.copy(x0)
    y = np.copy(y0)
    z = np.polyfit(x, y, 1)
    for ii in range(x.size):
        f = z[0]*x + z[1]
        res = y - f
        s = np.std(res)
        m = np.mean(res)
        ind = np.where(np.abs(res-m)>2*s)[0]
        if ind.size == 0: break
        ind = np.where(np.abs(res-m)<=2*s)[0]
        x = x[ind]
        y = y[ind]
        z = np.polyfit(x, y, 1)
    return z

def firstcal(uv, reds, pols=['x', 'y']):
    d0 = uv.data_array.reshape(uv.Ntimes, uv.Nbls, uv.Nfreqs, uv.Npols)
    f0 = uv.flag_array.reshape(uv.Ntimes, uv.Nbls, uv.Nfreqs, uv.Npols)
    a1 = uv.ant_1_array[:uv.Nbls]
    a2 = uv.ant_2_array[:uv.Nbls]
    bls = 128 * a1 + a2
    md = np.ma.masked_array(d0, f0)
    md = np.mean(md, axis=0)
    data = {}
    for r in reds:
        for b in r:
            i,j = b
            try:
                ind = np.where(bls==128*i+j)[0][0]
                data[b] = md[ind, :]
            except:
                ind = np.where(bls==128*j+i)[0][0]
                data[b] = md[ind, :].conj()
    pol2num = {'x': 0, 'y': 1}
    nbls, ants, a2n = cal_dim(reds)
    g0 = {}
    fq = uv.freq_array[0] / 1e9
    for p in pols:
        g0[p] = {}
        d = np.zeros(nbls)
        A = np.zeros((nbls, len(ants)))
        x = np.zeros(len(ants))
        n = 0
        for r in reds:
            for ii in range(len(r)-1):
                b1 = r[ii]
                d1 = data[b1][:,pol2num[p]]
                i, j = b1
                for jj in range(ii+1, len(r)):
                    b2 = r[jj]
                    d2 = data[b2][:,pol2num[p]]
                    k, l = b2
                    y = d1.conj()*d2
                    med = np.ma.median(np.abs(y))
                    ind = np.where(y.mask==False)[0]
                    ind2 = np.where(np.abs(y[ind]) > med)
                    tau = np.unwrap(np.angle(y[ind][ind2]))
                    z = adaptfit(fq[ind][ind2], tau)
                    d[n] = z[0]
                    A[n,a2n[i]] = -1
                    A[n,a2n[j]] = 1
                    A[n,a2n[k]] = 1
                    A[n,a2n[l]] = -1
                    n += 1
        x = np.linalg.pinv(A.transpose().dot(A),rcond=1e-8).dot(A.transpose()).dot(d)
        for ii in range(x.size):
            g0[p][ants[ii]] = np.complex64(np.exp(1j * x[ii] * fq))
    return g0
