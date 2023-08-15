# *********************************************************
# Author: Tiancheng Xiong
# Date: June 27, 2023
# Purpose: Plot the Cross section plot based on the input of data (data should be in specific format! check our sample_data.txt)
# Notice: 
#  *Adapt to python 3.11.3
#  *Feel free to change the settings of the code to customize the plot
# *********************************************************
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d
import math

def plot_cross_section(filePath):
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

    theta = np.linspace(thetaMin, thetaMax, nx)
    phi = np.linspace(phiMin, phiMax, ny)

    tg, pg = np.meshgrid(theta, phi)

    xlist = theta.tolist()
    ylist = phi.tolist()

    reverseYlist = [-i for i in ylist]

    vertical_mid = (9 + ny + 8)//2     # corresponds to Y
    horizontal_mid = (nx)//2    # corresponds to X

    # print(vertical_mid, horizontal_mid)

    # vertical_mid_data = None
    with open(filePath, 'r') as f:
        # Read all the lines in the file into a list
        lines = f.readlines()
        # print(lines)
        # Get the third line from the list (index 2)
        vertical_mid_data = lines[vertical_mid]
        # Split the line into a list using a delimiter (e.g. comma)
        vertical_mid_data_list = vertical_mid_data.strip().split()


        

    column_data = []
    counter = 0
    for line in lines:
        counter += 1
        if counter >= 9:
        # columns = line.split('\t')
        # if len(lines) != 0:
        #     columns = line.split('\t')
        #     column_data.append(columns[horizontal_mid])
            columns = line.split()
            columns = [float(i) for i in columns]
            # print(columns)
            column_data.append(float(columns[horizontal_mid+1]))
        # print(type(columns))
        # column_data.append(float(columns[horizontal_mid-1]))
    
    # print(column_data)
   
    vertical_mid_data_list = [float(val)/10000 for val in vertical_mid_data_list]

    column_data = [float(val)/10000 for val in column_data]

    # get the divergence 
    # f1 = interp2d(xlist, vertical_mid_data_list, np.arange(len(xlist)), kind="cubic")
    # y = 1 / (math.exp(2))
    # x = f1(np.unique(xlist), y)
    # print(x)

    # f1 = interp1d(vertical_mid_data_list, xlist, kind = 'quadratic')
    # y = 1 / (math.exp(2))
    # x = f1(y)
    # print(x)

   # get the divergence 
    y_val = max(vertical_mid_data_list) * (1 / (math.exp(2)))
    tempDf = pd.DataFrame({"x_column": xlist, "y_column":vertical_mid_data_list})
    ret = find_closest(tempDf, y_val)
    print(ret[0] - ret[1])

    tempDf = pd.DataFrame({"x_column": reverseYlist, "y_column":column_data})
    ret = find_closest(tempDf, y_val)
    print(ret[0] - ret[1])

  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(xlist, vertical_mid_data_list)
    ax1.grid(True)
    ax1.set_title("measure 1")
    ax1.set_xlabel("Divergence (deg)")
    ax1.set_ylabel("Irradiance (W/sr)")

    # ax1.annotate('x10000', xy=(0, 0.95), xycoords='axes fraction')
    fig.text(0.03, 0.96, '10^4', ha='center', va='center')
    fig.text(0.53, 0.96, '10^4', ha='center', va='center')
    
    # plt.plot(xlist, my_list)

    ax2.plot(reverseYlist, column_data)
    ax2.grid(True)
    ax2.set_title("measure 1")
    ax2.set_xlabel("Divergence (deg)")
    ax2.set_ylabel("Irradiance (W/sr)")
    # ax2.set_axisbelow(True)
    # ax2.set_xticks(xtk)
    # ax2.set_yticks(ytk)

    plt.tight_layout()
    plt.show()

def find_closest(df, y_value):
    df_positive = df[df['x_column'] > 0]
    df_negative = df[df['x_column'] < 0]
    closest_rows_positive = df_positive.iloc[(df_positive['y_column'] - y_value).abs().argsort()[:]]
    closest_rows_negative = df_negative.iloc[(df_negative['y_column'] - y_value).abs().argsort()[:]]
    return closest_rows_positive['x_column'].values[0], closest_rows_negative['x_column'].values[0]




# if __name__ == "__main__":
#     # plot_cross_section('sample_data.txt')
#     # U:\William\Ewok 3\MZ Evo Ewok 3 8Emitters\Conoscope\20230810155842_Measurement_Beam_Profile_MZ_LRA176_Test.txt
#     plot_cross_section('U:\William\Ewok 3\MZ Evo Ewok 3 8Emitters\Conoscope\Measurement_Beam_Profile_MZ_LRA176_Test.txt')
