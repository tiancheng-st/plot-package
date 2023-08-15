# *********************************************************
# Author: Tiancheng Xiong, Jaeyong Kim
# Date: June 26, 2023
# Purpose: Plot the beam profile based on the input of data (data should be in specific format! check our sample_data.txt)
# Notice: 
#  *You can change the type of cmap to get different color maps. "jet.png" used the jet style for cmap. "hsv"
#  *Adapt to python 3.11.3
#  *Feel free to change the settings of the code to customize the plot
#  *Preferred level value is 500 to 1000
# *********************************************************
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

# Define a function to format the tick labels
def format_func(value, tick_number):
    return int(value / 10000)

def convertRadiantToPolarMapAndPlot(filePath):
    """
    Converts a file with radiant data into a polar map
    Args: filePath: the path of data file

    Returns: None
        
    """
    # Read the file and extract header information
    # ---------------------------------------Data Clean--------------------------------------- 
    with open(filePath) as f:
        header = [f.readline() for _ in range(8)]

    nx = int(header[2][2:])     # get the number of columns
    ny = int(header[3][2:])     # get the number of rows
    # print(nx)
    # print(ny)

    thetaMin = float(header[4][4:])     # get the value of BegX
    thetaMax = float(header[5][4:])     # get the value of EndX
    # print(thetaMin)
    # print(thetaMax)

    phiMin = float(header[6][4:])       # get the value of BegY
    phiMax = float(header[7][4:])       # get the value of EndY
    # print(phiMin)
    # print(phiMax)

    # returns evenly spaced numbers over a specified interval 
    theta = np.linspace(thetaMin, thetaMax, nx)
    phi = np.linspace(phiMin, phiMax, ny)

    tg, pg = np.meshgrid(theta, phi)

    xlist = theta.tolist()
    ylist = phi.tolist()
    reverseYlist = [-i for i in ylist]
    # print(len(xlist))
    # print(len(ylist))

    # the data format for z should be a 2D array !!
    start_line = 8
    data = np.genfromtxt(filePath, skip_header=start_line)
    z = np.array(data)
    # print(z)

    # ---------------------------------------Graph Plot---------------------------------------
    fig, ax = plt.subplots()
    #invert y-axis
      #ax.set_ylim(ax.get_ylim()[::-1])
      #ax.invert_yaxis()
    # set the label and title
    ax.set_xlabel('Theta (deg)')
    ax.set_ylabel('Theta (deg)')
    ax.set_title("Measurement...")
    # adjust the ticks
    ax.set_xticks(np.arange(-40, 50, 10))
    ax.set_yticks(np.arange(-40, 50, 10))

    # Limit x-axis and y-axis
    ax.set_xlim ([-50, 50])
    ax.set_ylim ([-50, 50])

    # plot the graph
    contour = plt.contourf(xlist,reverseYlist,z,cmap="jet",levels = 500)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    cbar = plt.colorbar(contour)
    # set a notation for colorbar
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    cbar.ax.set_ylabel('10^4')
    cbar.ax.xaxis.set_label_position('top')

    
    plt.show()



# if __name__ == "__main__":
#     convertRadiantToPolarMapAndPlot('U:\William\Ewok 3\MZ Evo Ewok 3 8Emitters\Conoscope\Measurement_Beam_Profile_MZ_LRA176_Test.txt')