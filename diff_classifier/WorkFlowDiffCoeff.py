#!/usr/bin/env python3
import os
import glob
import numpy as np
import skimage.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import diff_classifier.msd as msd
import diff_classifier.features as ft
import diff_classifier.heatmaps as hm
import data_analysis as da
import argparse

# TODO: Remember Check the Velocity/Speed code

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", help="File name", required=True)
parser.add_argument("-p", "--param", help="Parameter type to plot eg. Brownian Motion Diffusion Coeff (BM), "
                                          "Diffusion Coeff (D), Mesh Size (M), Speed and Velocity (SV)", required=False)
parser.add_argument("-ds", "--dshape", help="Desired Shape", type=int, required=True)
parser.add_argument("-vt", "--video_time", help="Time of Video in seconds", type=int, required=True)
parser.add_argument("-r", "--resolution", help="resolution in px/micron", type=int, required=True)
parser.add_argument("-rs", "--rs", help="radius of the nanoparticle", type=int, required=False)
parser.add_argument("-rf", "--rf", help="the radius of the polymer chains", type=int, required=False)
parser.add_argument("-pt", '--plot_traj', help="plot the trajectories", default=False, action='store_true')
parser.add_argument("-msdp", '--mean_sqr_disp', help="Mean Square Displacement Plot", default=False,
                    action='store_true')
parser.add_argument("-hmp", '--heat_map', help="heat map plot", default=False, action='store_true')
parser.add_argument("-pap", '--precision_averaging', help="Precision Averaging Plot", default=False,
                    action='store_true')
parser.add_argument("-ph", '--histogram', help="Diffusion Coefficient Histogram", default=False, action='store_true')
parser.add_argument("-b", '--boxplot', help=" -Plot Boxplot", default=False, action='store_true')
parser.add_argument("-m", '--mesh_size', help="Mesh Size", default=False, action='store_true')
parser.add_argument('-l', '--list', nargs='+', help='parameters to plot with boxplot', required=False)

args = parser.parse_args()

# PARAMETERS
filename = args.file
ds = args.dshape
vid_time = args.video_time
resolution = args.resolution
pt = args.plot_traj
msdp = args.mean_sqr_disp
hmp = args.heat_map
pap = args.precision_averaging
ph = args.histogram
p = args.param
boxplot = args.boxplot
mesh = args.mesh_size
parameter_list = args.list
rs = args.rs
rf = args.rf

tiffname = filename + ".tif"
ft_file = 'features_{}.csv'.format(filename)
msd_file = 'msd_{}.csv'.format(filename)

os.chdir('/Users/claudialozano/Dropbox/PycharmProjects/AD_nanoparticle/diff_classifier/notebooks/development/MPT_Data'
         '/{}/'.format(filename))

print("Loading Video")
ovideo = sio.imread(tiffname)  # Read tif file
oshape = ovideo.shape  # Shape of Original Video
dshape = (ds, ds)  # Desired shape of smaller videos

umppx = 1 / resolution  # in microns per pixel
fps = oshape[0] / vid_time

# Splitting original video into smaller videos sized
nvideo = np.zeros(oshape, dtype=ovideo.dtype)  # Create an empty array with original's video shape
nvideo[0:oshape[0], 0:oshape[1], :] = ovideo

# Create new empty array with desired shape dimensions
new_image = np.zeros((oshape[0], dshape[0], dshape[1]), dtype=ovideo.dtype)
names = []
division_rows = int(oshape[1] / dshape[0])  # Determine the number of row we want to the divide the new video into
division_cols = int(oshape[2] / dshape[1])  # Determine the number of columns we want to the divide the new video into

for row in range(division_rows):
    for col in range(division_cols):
        new_image = nvideo[:, row * dshape[0]:(row + 1) * dshape[0], col * dshape[1]:(col + 1) * dshape[1]]
        current = tiffname.split('.tif')[0] + '_%s_%s.tif' % (row, col)
        sio.imsave(current, new_image)
        names.append(current)

print("Loading Trajectory Files")
files = glob.glob('*Traj*')  # get rename the trajectory .csv files

# Defining list of all small videos to be quantified (calculating msds and features)
names = []
length = oshape[0]  # number of frames
for i in range(0, division_rows):
    for j in range(0, division_cols):
        names.append('{}_{}_{}.tif'.format(filename, i, j))

# Calculating MSDs and features for the videos that have been tracked. This uses the Traj .csv files and generates an
# msd and features .csv file for the videos being quantified. It uses the multiple small vid Traj .csv files to generate
# a single msd and features .csv file. Essentially, this runs the kn.assemble_msds() function

"""Creating MSD File"""
print("Creating MSD File")

counter = 0
merged = False
for name in names:
    try:
        # Get the numbering of the sectioned videos ####
        row = int(name.split(filename)[1].split('.')[0].split('_')[1])
        col = int(name.split(filename)[1].split('.')[0].split('_')[2])

        local_name = "{}_{}_{}.csv".format("Traj", row, col)

        to_add = pd.read_csv(local_name)  # Gets all the values from the tajectory file

        # Mine have an extra 3 rows of non important information so we get rid of the first three rows
        to_add = to_add.iloc[3:]

        # Deleting unecessary columns in the CSV file that ImageJ(FIJI) outputs #####
        del to_add['LABEL']
        del to_add['POSITION_Z']
        del to_add['POSITION_T']
        del to_add['RADIUS']
        del to_add['VISIBILITY']
        del to_add['MANUAL_SPOT_COLOR']
        del to_add['MEDIAN_INTENSITY_CH1']
        del to_add['MIN_INTENSITY_CH1']
        del to_add['MAX_INTENSITY_CH1']
        del to_add['TOTAL_INTENSITY_CH1']
        del to_add['STD_INTENSITY_CH1']
        del to_add['CONTRAST_CH1']
        del to_add['ID']  # Not really used by msds2 file so I also delete it

        # Get the names and format in the one taken by the MSDS2 function
        to_add = to_add.rename(columns={'TRACK_ID': 'Track_ID', 'QUALITY': 'Quality', 'FRAME': 'Frame',
                                        'POSITION_X': 'X', 'POSITION_Y': 'Y',
                                        'MEAN_INTENSITY_CH1': 'Mean_Intensity', 'SNR_CH1': 'SN_Ratio'})

        to_add.sort_values(['Track_ID', 'Frame'], ascending=[1, 1])
        to_add = to_add.astype('float64')  # Declaring the values to have decimal points

        counter = 0
        partids = to_add.Track_ID.unique()  # Getting all the unique TRACK_IDs: Unique particles
        for partid in partids:
            to_add.loc[to_add.Track_ID == partid, 'Track_ID'] = counter
            counter = counter + 1

        # Taking into account that we are in a different row and column of the original video
        # Since we divided the videos the X and Y positions are offset and this takes care of that

        to_add['X'] = to_add['X'] + dshape[0] * col
        to_add['Y'] = dshape[1] - to_add['Y'] + dshape[1] * (division_rows - 1 - row)

        # Cheking if it is the first video, if so we need to create merge array
        if counter == 0 or type(merged) == bool:
            print('counter is 0')
            merged = msd.all_msds2(to_add, frames=length)

        # If merged has already been created we can concatenate the next array
        else:
            # Since each particle is different of the different videos we need to take that into account.
            # eg Particle with Track ID 1 from video 1_1 is not the same as Particle with Track ID 1 from
            # video 1_2, so we shift the Track IDs
            if merged.shape[0] > 0:
                print('merged.shape is greater than 0')
                to_add['Track_ID'] = to_add['Track_ID'] + max(merged['Track_ID']) + 1
            else:
                print('else')
                to_add['Track_ID'] = to_add['Track_ID']

            print('concat')
            msds2 = msd.all_msds2(to_add, frames=length)
            merged = pd.concat([merged, msds2], axis=0, join='outer')

        counter = counter + 1
        print('Done calculating MSDs for row {} and col {}'.format(row, col))

    except pd.errors.EmptyDataError:

        print('Found empty file : {}'.format(name))

print("Finished Merging")

merged.to_csv(msd_file)  # Create CSV file from the array containing the MSDs
merged_ft = ft.calculate_features(merged)  # Create Feature file
merged_ft.to_csv(ft_file)   # Save features in to a CSV

# If parameters is SV speed velocity
if p == "SV":
    sv = da.features_from_velocity(names, filename, umppx, fps, vid_time)

# Plot the Trajectories
if pt:
    print("Plotting Trajectories")
    hm.plot_trajectories(filename, resolution=dshape[1], rows=division_rows, cols=division_cols,
                         upload=False, figsize=(12, 12))

# Plot Mean Square Displacement Plot
if msdp:
    print('Plotting the Mean Square')
    geomean, geoSEM = hm.plot_individual_msds(filename, x_range=4, y_range=15, umppx=umppx, fps=fps, upload=False)

# Plot Number of particles per frame
if hmp:
    print("Plotting the heatmap")
    hm.particles_in_frame(filename, x_range=oshape[0] + 10, y_range=50)

# Plots pre-calculated precision-weighted averages of MSD datasets calculated from precision_averaging
if pap:
    print("Plotting the re-calculated precision-weighted averages of MSD datasets ")
    geo_stder = {}
    geomean = {}

    compared = [filename]
    for fl in compared:
        geomean_array, geo_stder_array = msd.geomean_msdisp(filename, umppx=umppx, fps=fps, upload=False)

        geo_stder[fl] = geo_stder_array
        geomean[fl] = geomean_array

    weights, w_holder = msd.precision_weight(compared, geo_stder)

    geodata = msd.precision_averaging(compared, geomean, geo_stder, weights, save=False)

    msd.plot_all_experiments(compared, yrange=(10 ** -1, 10 ** 3), fps=fps, xrange=(10 ** -1, 10 ** 1.5), upload=False,
                             outfile='precision-weighted averages of MSD datasets.png', exponential=True, labels=None,
                             log=True)
    plt.savefig('precision-weighted averages of MSD datasets.png', bbox_inches='tight')

if ph:
    print("Plotting the Histogram")
    hm.plot_histogram(filename, xlabel='Log Diffusion Coefficient Dist', ylabel='Trajectory Count',
                      fps=fps, umppx=umppx, frames=oshape[0], y_range=30, frame_interval=20, frame_range=100,
                      analysis='log', theta='D', upload=False)
if boxplot:
    print("Plotting the boxplot")
    da.boxplot_feature(filename, col_list=parameter_list, file=p, umppx=umppx, fps=fps)

if mesh:
    print("Calculating Mesh Size")
    da.pore_size(filename, rs, rf, DC="DiffCoeffBM", umppx=umppx)
