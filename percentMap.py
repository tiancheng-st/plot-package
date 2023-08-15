# import radiantToPolar as pm
# import makeCircle as mc
# import normalise2sum as normal
# import makeGauss as mg
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os


def makeCircle(r):
    if r >= 1:
        x = np.arange(0, np.ceil(2*r)+1) - np.ceil(2*r+1)/2 + 0.5
        y = np.arange(0, np.ceil(2*r)+1) - np.ceil(2*r+1)/2 + 0.5
        xg, yg = np.meshgrid(x, y)
        dmap = np.sqrt(xg**2 + yg**2)
        out = np.zeros((xg.shape[0], xg.shape[1]))
        out[dmap <= r] = 1
        idx = np.logical_and(dmap > r, dmap <= r+1)
        out[idx] = 1 - (dmap[idx] - r)
    elif r < 1 and r > 0:
        out = 1
    else:
        out = 0
    return out

def normalise2Max(m):
    m = np.double(m)
    maximum = np.max(m)
    m = m / (np.sqrt(maximum * np.conj(maximum)))
    return m


def makeGauss(MapSize, sig):
    origin = [np.ceil(MapSize[0]/2), np.ceil(MapSize[1]/2)]
    if len(sig) == 1:
        sig.append(sig[0])
    if MapSize[0] == 1:
        origin[0] = 0
    if MapSize[1] == 1:
        origin[1] = 0
    # print()
    # data type transform
    size0 = int(MapSize[0])
    size1 = int(MapSize[1])
    f = np.zeros((size0, size1))
    for i in range(size0):
        for j in range(size1):
            y = i - origin[0]
            x = j - origin[1]
            f[i,j] = 1/(np.sqrt(2*np.pi*sig[0])) * 1/(np.sqrt(2*np.pi*sig[1])) * \
                     np.exp(-((x)**2)/(2*sig[0]**2) - ((y)**2)/(2*sig[1]**2))
    f = normalise2Max(f)
    return f


def normalise2sum(m):
    ii = np.where(np.isinf(m))
    jj = np.where(np.isnan(m))
    idxBad = np.concatenate((ii, jj))
    m[idxBad] = 0
    summing = np.sum(m)
    m = m / np.sqrt(summing * np.conj(summing))
    if np.sum(np.isnan(m)) >= 1:
        temp = np.ones((m.shape[0], m.shape[1])) / np.size(m)
        m = temp
        del temp
    return m

def convertRadiantToPolarMap(radiantFileName):
    headerlines = 8
    with open(radiantFileName,'r') as fid:
        header = [next(fid) for x in range(headerlines)]
        nx = int(header[2][2:])
        ny = int(header[3][2:])
        # print(nx, ny)
        thetaMin = float(header[4][5:])
        thetaMax = float(header[5][5:])
        phiMin = float(header[6][5:])
        phiMax = float(header[7][5:])
        theta = np.linspace(thetaMin,thetaMax,nx)
        phi = np.linspace(phiMin,phiMax,ny)
        tg,pg = np.meshgrid(theta,phi)

        print(len(tg), len(pg))
        intensityData = np.loadtxt(radiantFileName,skiprows=8)
        # intensityData = intensityData[:,:-1]

        print(intensityData.size)

        polarMap = np.zeros((ny,nx,3))
        polarMap[:,:,0] = intensityData
        polarMap[:,:,1] = tg
        polarMap[:,:,2] = pg
    return polarMap

def run_percentageMap(filename):
    polarMap = convertRadiantToPolarMap(filename)
    angPerPix = 0.0466
    pmap_data = polarMap   
    LS_Map = pmap_data[:, :, 0]
    kernelRadius  = 4.01/2 / angPerPix
    GaussRadius	= 1/angPerPix
    kernel = makeCircle(kernelRadius)
    # print("kernel: ", kernel)
    TotalPower = sum(sum(LS_Map))
    print("TotalPower: ", TotalPower)
    # mg.makeGauss()
    # normal.normalise2sum()

    Cut = 40
    res = 1 / (pmap_data[0, 1, 1] - pmap_data[0, 0, 1])
    R0 = Cut*res
    center = np.ceil(np.size(pmap_data[:, :, 0], 0) / 2)

    # convolution
    temp = normalise2sum(makeGauss(np.round([2*GaussRadius, 2*GaussRadius]), [GaussRadius/1.5]))
    LS_Map = convolve2d(LS_Map, temp, mode="same")
    SafetyMap = convolve2d(LS_Map, kernel, mode="same")
    # ii, jj = np.where(SafetyMap == np.max(SafetyMap))
    # peakPos = np.array([ii[0], jj[0]])

    maxMo = np.max(SafetyMap)

    # Compute output intensity gain modMap
    PercentMap = 100 * SafetyMap / TotalPower
    MaxPercent = 100 * maxMo / TotalPower
    # plot
    # Define variables
    savefilename = "test"
    FoV = f'{savefilename}PercentMap_{str(MaxPercent)}'
    FoV = FoV.replace('.', ',')
    resultsDir = os.path.join(os.getcwd(), FoV)

    fig, ax = plt.subplots()
    im = ax.imshow(PercentMap, cmap = "jet")
    ax.set_xlim([center-(R0+5*res), center+(R0+5*res)])


    # ax.set_ylim([center-(R0+5*res), center+(R0+5*res)])
    # flip the y-axis
    ax.set_ylim([center+(R0+5*res), center-(R0+5*res)])

    fig.colorbar(im)
    ax.set_box_aspect(1)
    ax.grid(True, color='w')
    ax.set_xticks([center+j*res for j in range(-70, 71, 10)])
    ax.set_xticklabels([j for j in range(-70, 71, 10)])
    ax.set_yticks([center+j*res for j in range(70, -71, -10)])
    ax.set_yticklabels([j for j in range(70, -71, -10)])
    ax.set_title(FoV, fontsize=12)
    ax.set_xlabel('Theta (deg)', fontsize=12)
    ax.set_ylabel('Theta (deg)', fontsize=12)

    # plt.savefig(resultsDir+'.png', dpi=300)
    # plt.savefig(resultsDir+'.svg', dpi=300)
    # plt.close(fig)
    plt.show()

    return PercentMap


if __name__ == "__main__":
    temp = run_percentageMap("U:\William\Ewok 3\MZ Evo Ewok 3 8Emitters\Conoscope\_500ps\Measurement_Beam_Profile_MZ_LRA135_500ps.txt")
    print(temp)


