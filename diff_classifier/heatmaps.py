import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import scipy.stats as stats
import os

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy.ma as ma
import matplotlib.cm as cm
import diff_classifier.aws as aws
import random
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import math
from sklearn import preprocessing
from PIL import Image
import glob
import itertools


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    counter = 0
    for p1, region in enumerate(vor.point_region):
        try:
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())
        except KeyError:
            counter = counter + 1
            # print('Oops {}'.format(counter))

    return new_regions, np.asarray(new_vertices)


def plot_heatmap(prefix, feature='asymmetry1', vmin=0, vmax=1, resolution=512, rows=4, cols=4,
                 upload=True, dpi=None, figsize=(12, 10), remote_folder="01_18_Experiment",
                 bucket='ccurtis.data'):
    """
    Plot heatmap of trajectories in video with colors corresponding to features.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    feature: string
        Feature to be plotted.  See features_analysis.py
    vmin: float64
        Lower intensity bound for heatmap.
    vmax: float64
        Upper intensity bound for heatmap.
    resolution: int
        Resolution of base image.  Only needed to calculate bounds of image.
    rows: int
        Rows of base images used to build tiled image.
    cols: int
        Columns of base images used to build tiled images.
    upload: boolean
        True if you want to upload to s3.
    dpi: int
        Desired dpi of output image.
    figsize: list
        Desired dimensions of output image.

    Returns
    -------

    """
    # Inputs
    # ----------

    merged_ft = pd.read_csv(prefix)
    string = feature
    leveler = merged_ft[string]
    t_min = vmin
    t_max = vmax
    ires = resolution

    # Building points and color schemes
    # ----------
    zs = ma.masked_invalid(merged_ft[string])
    zs = ma.masked_where(zs <= t_min, zs)
    zs = ma.masked_where(zs >= t_max, zs)
    to_mask = ma.getmask(zs)
    zs = ma.compressed(zs)

    xs = ma.compressed(ma.masked_where(to_mask, merged_ft['X'].astype(int)))
    ys = ma.compressed(ma.masked_where(to_mask, merged_ft['Y'].astype(int)))
    points = np.zeros((xs.shape[0], 2))
    points[:, 0] = xs
    points[:, 1] = ys
    vor = Voronoi(points)

    # Plot
    # ----------
    fig = plt.figure(figsize=figsize, dpi=dpi)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    my_map = cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(t_min, t_max, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    test = 0
    p2 = 0
    counter = 0
    for i in range(0, points.shape[0] - 1):
        try:
            polygon = vertices[regions[p2]]
            point1 = Point(points[test, :])
            poly1 = Polygon(polygon)
            check = poly1.contains(point1)
            if check:
                plt.fill(*zip(*polygon), color=my_map(norm(zs[test])), alpha=0.7)
                p2 = p2 + 1
                test = test + 1
            else:
                p2 = p2
                test = test + 1
        except IndexError:
            print('Index mismatch possible.')

    mapper.set_array(10)
    plt.colorbar(mapper)
    plt.xlim(0, ires * cols)
    plt.ylim(0, ires * rows)
    plt.axis('off')

    print('Plotted {} heatmap successfully.'.format(prefix))
    outfile = 'hm_{}_{}.png'.format(feature, prefix)
    fig.savefig(outfile, bbox_inches='tight')
    if upload == True:
        aws.upload_s3(outfile, remote_folder + '/' + outfile, bucket_name=bucket)


def plot_scatterplot(prefix, feature='asymmetry1', vmin=0, vmax=1, resolution=512, rows=4, cols=4,
                     dotsize=10, figsize=(12, 10), upload=True, remote_folder="01_18_Experiment",
                     bucket='ccurtis.data'):
    """
    Plot scatterplot of trajectories in video with colors corresponding to features.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    feature: string
        Feature to be plotted.  See features_analysis.py
    vmin: float64
        Lower intensity bound for heatmap.
    vmax: float64
        Upper intensity bound for heatmap.
    resolution: int
        Resolution of base image.  Only needed to calculate bounds of image.
    rows: int
        Rows of base images used to build tiled image.
    cols: int
        Columns of base images used to build tiled images.
    upload: boolean
        True if you want to upload to s3.

    """
    # Inputs
    # ----------
    merged_ft = pd.read_csv(prefix)
    string = feature
    leveler = merged_ft[string]
    t_min = vmin
    t_max = vmax
    ires = resolution

    norm = mpl.colors.Normalize(t_min, t_max, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    zs = ma.masked_invalid(merged_ft[string])
    zs = ma.masked_where(zs <= t_min, zs)
    zs = ma.masked_where(zs >= t_max, zs)
    to_mask = ma.getmask(zs)
    zs = ma.compressed(zs)
    xs = ma.compressed(ma.masked_where(to_mask, merged_ft['X'].astype(int)))
    ys = ma.compressed(ma.masked_where(to_mask, merged_ft['Y'].astype(int)))

    fig = plt.figure(figsize=figsize)
    plt.scatter(xs, ys, c=zs, s=dotsize)
    mapper.set_array(10)
    plt.colorbar(mapper)
    plt.xlim(0, ires * cols)
    plt.ylim(0, ires * rows)
    plt.axis('off')

    print('Plotted {} scatterplot successfully.'.format(prefix))
    outfile = 'scatter_{}_{}.png'.format(feature, prefix)
    fig.savefig(outfile, bbox_inches='tight')
    if upload == True:
        aws.upload_s3(outfile, remote_folder + '/' + outfile, bucket_name=bucket)


def plot_trajectories(prefix, resolution=512, rows=4, cols=4, upload=True,
                      remote_folder="01_18_Experiment", bucket='ccurtis.data',
                      figsize=(12, 12), subset=True, size=1000):
    """
    Plot trajectories in video.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    resolution: int
        Resolution of base image.  Only needed to calculate bounds of image.
    rows: int
        Rows of base images used to build tiled image.
    cols: int
        Columns of base images used to build tiled images.
    upload: boolean
        True if you want to upload to s3.

    """
    merged = pd.read_csv('msd_{}.csv'.format(prefix))
    particles = int(max(merged['Track_ID']))
    if particles < size:
        size = particles - 1
    else:
        pass
    particles = np.linspace(0, particles, particles - 1).astype(int)
    if subset:
        particles = np.random.choice(particles, size=size, replace=False)
    ires = resolution

    fig = plt.figure(figsize=figsize)
    for part in particles:
        x = merged[merged['Track_ID'] == part]['X']
        y = merged[merged['Track_ID'] == part]['Y']
        plt.plot(x, y, color='k', alpha=0.7)

    plt.xlim(0, ires * cols)
    plt.ylim(0, ires * rows)
    plt.axis('off')

    print('Plotted {} trajectories successfully.'.format(prefix))
    outfile = 'traj_{}.png'.format(prefix)
    fig.savefig(outfile, bbox_inches='tight')
    if upload:
        aws.upload_s3(outfile, remote_folder + '/' + outfile, bucket_name=bucket)


def plot_histogram(prefix, xlabel='Log Diffusion Coefficient Dist', ylabel='Trajectory Count',
                   fps=100.02, umppx=0.16, frames=651, y_range=100, x_range=4, frame_interval=10, frame_range=50,
                   analysis='log', theta='D', ):
    """
    Plot heatmap of trajectories in video with colors corresponding to features.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    xlabel: string
        X axis label.
    ylabel: string
        Y axis label.
    fps: float64
        Frames per second of video.
    umppx: float64
        Resolution of video in microns per pixel.
    frames: int
        Number of frames in video.
    y_range: float64 or int
        Desire y range of graph.
    frame_interval: int
        Desired spacing between MSDs/Deffs to be plotted.
    analysis: string
        Desired output format.  If log, will plot log(MSDs/Deffs)
    theta: string
        Desired output.  D for diffusion coefficients.  Anything else, MSDs.
    upload: boolean
        True if you want to upload to s3.

    """
    # load data
    merged = pd.read_csv('msd_{}.csv'.format(prefix))
    frame_range = range(frame_interval, frame_range + frame_interval, frame_interval)

    # generate keys for legend
    bar = {}
    keys = []
    entries = []

    for i in range(0, len(list(frame_range))):
        keys.append(i)
        entries.append(str(10 * frame_interval * (i + 1)) + 'ms')

    set_x_limit = True
    set_y_limit = True

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure(figsize=(16, 6))

    counter = 0
    tt = []
    for i in frame_range:
        toi = i / fps  #### ASUMMING BROWIAN MOTION!!!!!!
        if theta == "MSD":
            factor = 1
        else:
            factor = 4 * toi

        if analysis == 'log':
            dist = np.log(umppx * umppx * merged.loc[merged.Frame == i, 'MSDs'].dropna() / factor)
            test_bins = np.linspace(-5, 5, 76)

        else:
            dist = umppx * umppx * merged.loc[merged.Frame == i, 'MSDs'].dropna() / factor
            tt.extend(dist.tolist())
            test_bins = np.linspace(0, 20, 76)

        histogram, test_bins = np.histogram(dist, bins=test_bins)

        # Plot_general_histogram_code
        avg = np.mean(dist)

        plt.rc('axes', linewidth=2)
        plot = histogram
        bins = test_bins
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        bar[keys[counter]] = plt.bar(center, plot, align='center', width=width, color=colors[counter],
                                     label=entries[counter])

        plt.xticks(test_bins.round(3) - .5, test_bins.round(3))
        plt.xticks(fontsize=5, rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.axvline(avg, color=colors[counter])
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel(ylabel, fontsize=30)

        counter = counter + 1
        if set_y_limit:
            plt.gca().set_ylim([0, y_range])

        if set_x_limit:
            plt.gca().set_xlim([0, x_range])

        plt.legend(fontsize=20, frameon=False)

    df = pd.DataFrame({'DiffCoeffBM': tt})
    df.to_csv('DiffCoeffBM_{}.csv'.format(prefix), index=False)
    outfile = 'hist_{}.png'.format(prefix)
    fig.savefig(outfile, bbox_inches='tight')
    plt.show()


def plot_particles_in_frame(prefix, x_range=600, y_range=2000, upload=True,
                            remote_folder="01_18_Experiment", bucket='ccurtis.data'):
    """
    Plot number of particles per frame as a function of time.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    x_range: float64 or int
        Desire x range of graph.
    y_range: float64 or int
        Desire y range of graph.
    upload: boolean
        True if you want to upload to s3.

    """
    merged = pd.read_csv('msd_{}.csv'.format(prefix))
    frames = int(max(merged['Frame']))
    framespace = np.linspace(0, frames, frames)
    particles = np.zeros((framespace.shape[0]))
    for i in range(0, frames):
        particles[i] = merged.loc[merged.Frame == i, 'MSDs'].dropna().shape[0]

    fig = plt.figure(figsize=(5, 5))
    plt.plot(framespace, particles, linewidth=4)
    plt.xlim(0, x_range)
    plt.ylim(0, y_range)
    plt.xlabel('Frames', fontsize=20)
    plt.ylabel('Particles', fontsize=20)

    outfile = 'in_frame_{}.png'.format(prefix)
    fig.savefig(outfile, bbox_inches='tight')
    if upload == True:
        aws.upload_s3(outfile, remote_folder + '/' + outfile, bucket_name=bucket)


def plot_individual_msds(prefix, x_range=100, y_range=20, umppx=0.16, fps=100.02, alpha=0.1, folder='.',
                         figsize=(10, 10), subset=True,
                         size=1000,
                         dpi=300):
    """
    Plot MSDs of trajectories and the geometric average.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    x_range: float64 or int
        Desire x range of graph.
    y_range: float64 or int
        Desire y range of graph.
    fps: float64
        Frames per second of video.
    umppx: float64
        Resolution of video in microns per pixel.
    alpha: float64
        Transparency factor.  Between 0 and 1.
    upload: boolean
        True if you want to upload to s3.

    Returns
    -------
    geo_mean: numpy array
        Geometric mean of trajectory MSDs at all time points.
    geo_SEM: numpy array
        Geometric standard errot of trajectory MSDs at all time points.

    """

    merged = pd.read_csv('msd_{}.csv'.format(prefix))

    fig = plt.figure(figsize=figsize)
    particles = int(max(merged['Track_ID']))

    if particles < size:
        size = particles - 1
    else:
        pass

    frames = int(max(merged['Frame']))

    x = merged['X'].values.reshape((particles + 1, frames + 1)) / fps

    particles = np.linspace(0, particles, particles - 1).astype(int)
    if subset:
        particles = np.random.choice(particles, size=size, replace=False)

    y = np.zeros((particles.shape[0], frames + 1))
    for idx, val in enumerate(particles):
        y[idx, :] = merged.loc[merged.Track_ID == val, 'MSDs'] * umppx * umppx
        x = merged.loc[merged.Track_ID == val, 'Frame'] / fps
        plt.plot(x, y[idx, :], 'k', alpha=alpha)

    geo_mean = np.nanmean(ma.log(y), axis=0)
    geo_SEM = stats.sem(ma.log(y), axis=0, nan_policy='omit')
    plt.plot(x, np.exp(geo_mean), 'k', linewidth=4)
    plt.plot(x, np.exp(geo_mean - geo_SEM), 'k--', linewidth=2)
    plt.plot(x, np.exp(geo_mean + geo_SEM), 'k--', linewidth=2)
    plt.xlim(0, x_range)
    plt.ylim(0, y_range)
    plt.xlabel('Tau (s)', fontsize=25)
    plt.ylabel(r'Mean Squared Displacement ($\mu$m$^2$)', fontsize=25)

    outfile = '{}/msds_{}.png'.format(folder, prefix)
    outfile2 = '{}/geomean_{}.csv'.format(folder, prefix)
    outfile3 = '{}/geoSEM_{}.csv'.format(folder, prefix)
    fig.savefig(outfile, bbox_inches='tight', dpi=dpi)
    np.savetxt(outfile2, geo_mean, delimiter=",")
    np.savetxt(outfile3, geo_SEM, delimiter=",")

    return geo_mean, geo_SEM


##########################################################
# Added Functions to better understand feature file
##########################################################

def plot_individual_features(*args, col_list=["alpha"],
                             gen_dir="/Users/claudialozano/Dropbox/PycharmProjects/AD_nanoparticle/diff_classifier/notebooks/development/MPT_Data/"):
    """
    Plots the individual feature values for each track ID.
    Can be used for a single feature file, or to plot different

    Parameters
    ----------
    :param *args : String
        Name of the file that we are getting the features from
    :param gen_dir:
        Name of the directory where all the folders with the files are stored

    Returns
    -------
    None: Plots features
    """
    col_list.append('Track_ID')
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    tt = pd.read_csv("{}{}/features_{}.csv".format(gen_dir, args[0], args[0]), usecols=col_list)
    features = list(tt.columns)  # gets the name of the features in the feature fill

    for feature in features:  # Creating a new plot for each feature
        for file in args:
            df = pd.read_csv("{}{}/features_{}.csv".format(gen_dir, file, file), usecols=col_list)
            if list(df[feature]):  # Making sure that the feature has data points
                rgb = (random.random(), random.random(),
                       random.random())  # Generate a random color for diff feature files
                plt.scatter(df['Track_ID'], df[feature],
                            marker=next(marker), color=rgb)  # Plot feature for each particle
            else:
                print('There has been an exception with:{}'.format(feature))

        plt.xlabel('Track_ID')
        plt.ylabel(feature)
        plt.title(feature)
        plt.tight_layout()
        if len(args) == 1:
            plt.savefig("{}/{}_plot_individual_features".format(args[0], feature))  # Saving Figure
        plt.savefig("{}_plot_individual_features".format(feature))  # Saving Figure
        plt.close()  # Clears up the figure for not overlapping points


def calc_min_cost_distance(calc, compare, feature):
    """
    Calculating the minimum vertical distance for a pair of points. Currently using the Eucledian Distance as the
    cost for the Cost Matrix

    Parameters
    ----------
    :param calc: feature data points
    :param compare: feature data points
    :param feature: desired feature that we are comparing

    Returns
    ----------
    :return: assigment of pairs that minimized the vertical distance
    """

    zeros = [0] * len(list(calc))  # Create Zeros list to disregard Track ID
    df = np.array([list(ele) for ele in list(zip(zeros, calc))])  # Create "Coordinates" Tuples
    df_compare = np.array([list(ele) for ele in list(zip(zeros, compare))])
    # Calculates the distances of the Y values as our Cost Matrix. Currently set as the Euclidean Distance
    spatial_distance = cdist(df, df_compare)
    # Searches for the pair of coordinates that decreases Costs
    row_ind, assignment = linear_sum_assignment(spatial_distance)
    return assignment, spatial_distance


def plot_compare_individual_features(prefix, compare_to,
                                     gen_dir="/Users/claudialozano/Dropbox/PycharmProjects/AD_nanoparticle/diff_classifier/notebooks/development/MPT_Data/"):
    """
    Mapping elements of one set to points to the elements of another set of points, such that the sum vertical
    distance is minimized.

    Parameters
    ----------
    :param prefix: Name of feature file
    :param compare_to: Name of file that we want to compare to
    :param gen_dir: Name of the directory where all the folders with the files are stored

    Returns
    --------
    :return: None
    Saves Plots

    """
    # Read csv files
    calc = pd.read_csv('{}{}/features_{}.csv'.format(gen_dir, prefix, prefix))
    compare = pd.read_csv('{}{}/features_{}.csv'.format(gen_dir, compare_to, compare_to))

    features = list(calc.columns)  # gets the name of the features in the feature fill

    for feature in features:  # for each feature in the file
        if not all(i != i for i in list(calc[feature])):  # Making sure that the feature has data points
            if not all(i != i for i in list(compare[feature])):
                compare_l = list(compare[feature])
                calc_l = list(calc[feature])
                # Gertting rid of NaN and Infinite values
                for i in range(len(calc_l)):

                    if math.isnan(calc_l[i]):
                        calc_l[i] = 0

                    if not np.isfinite(calc[feature][i]):
                        calc_l[i] = 0

                for i in range(len(compare_l)):
                    if not np.isfinite(compare_l[i]):
                        compare_l[i] = 0

                    if math.isnan(compare_l[i]):
                        compare_l[i] = 0
                assignment, distances = calc_min_cost_distance(calc_l, compare_l, feature)  # Get pair assignments

                # Generate coordinate tuples
                df = np.array([list(ele) for ele in list(zip(calc["Track_ID"], calc[feature]))])
                df_compare = np.array([list(ele) for ele in list(zip(compare["Track_ID"], compare[feature]))])

                N = min(df.shape[0], df_compare.shape[0])
                # Plot points
                plt.plot(df[:N, 0], df[:N, 1], 'bo')
                plt.plot(df_compare[:N, 0], df_compare[:N, 1], 'rs')

                for point in range(N):  # Plot the lines connecting the pair points
                    try:
                        print(df[point, 0], df_compare[assignment[point], 0])
                        print('tt', df[point, 1], df_compare[assignment[point], 1])
                        plt.plot([df[point, 0], df_compare[assignment[point], 0]],
                                 [df[point, 1], df_compare[assignment[point], 1]], 'k')
                    except:
                        print("Can't plot")

                plt.xlabel('Track_ID')
                plt.ylabel(feature)
                plt.title(feature)
                plt.tight_layout()
                os.mkdir("{}_vs_{}".format(prefix, compare_to))
                plt.savefig("{}_vs_{}/{}_plot_compare_individual_features".format(prefix, compare_to, feature))
                plt.close()


            else:
                print('There has been an exception with:{}'.format(feature))
        else:
            print('There has been an exception with:{}'.format(feature))


def plot_compare_features(prefix, compare_to, name_file,
                          gen_dir="/Users/claudialozano/Dropbox/PycharmProjects/AD_nanoparticle/diff_classifier/notebooks/development/MPT_Data/"):
    """
    Creates a bar plot that determines a difference factor for each of the different features.
    The difference factor ranges from 0-1 where a difference of 0 means that they are the same.
    The difference is calculated is the best case scenario difference.
    Each of the particles in one file is paired to the particle that most closely resembles its features in the other
    video. Such that the sum of the distance between the particles' features is minimized.
    Only the vertical distance was taken into account since the Track ID is not necessarily a feature of the particle
    and could be randomly assigned to the different particles by TrackMate.

    Parameters
    ----------
    :param prefix: String
        Name of file that we want to compare
    :param compare_to: String
        Name of file that we want to compare

    Return
    -------
    :return: Plot and Directory
        Saves the bar plot of the difference of the features as well as a directory mapping feature to difference
        factor
    """

    # Read csv files
    calc = pd.read_csv('{}{}/features_{}.csv'.format(gen_dir, prefix, prefix))
    compare = pd.read_csv('{}{}/features_{}.csv'.format(gen_dir, compare_to, compare_to))

    features = list(compare.columns)  # gets the name of the features in the feature fill

    diff_factor = {}  # Initialize directory

    # normalize values in a range of 0-1 so that the magnitude difference between features does not affect
    # the difference factor

    for feature in features:

        if not all(i != i for i in list(calc[feature])):  # Making sure that the feature has data points
            if not all(i != i for i in list(compare[feature])):
                distance = 0  # Re-initializes distance between features
                compare_l = list(compare[feature])
                calc_l = list(calc[feature])
                # Getting rid of NaN and infinite values
                for i in range(len(calc_l)):

                    if math.isnan(calc_l[i]):
                        calc_l[i] = 0

                    if not np.isfinite(calc[feature][i]):
                        calc_l[i] = 0

                for i in range(len(compare_l)):
                    if not np.isfinite(compare_l[i]):
                        compare_l[i] = 0

                    if math.isnan(compare_l[i]):
                        compare_l[i] = 0

                normalized_df = preprocessing.normalize([np.array(calc_l)], norm='max')
                normalized_df = normalized_df.tolist()

                normalized_df_compare = preprocessing.normalize(np.array([compare_l]), norm="max")
                normalized_df_compare = normalized_df_compare.tolist()
                # Generate coordinate tuples
                df = np.array([list(ele) for ele in list(zip(calc["Track_ID"], normalized_df[0]))])
                df_compare = np.array(
                    [list(ele) for ele in list(zip(compare["Track_ID"], normalized_df_compare[0]))])

                # Get pair assignments

                assignment, distances = calc_min_cost_distance(calc_l, compare_l, feature)
                N = len(assignment)
                for point in range(N):
                    # Add the distances between all the points
                    distance += abs(df[point, 1] - df_compare[assignment[point], 1])

                try:
                    penalty = 0  # Penalty for tracking more or less particles
                    diff_factor[feature] = (distance / N) + penalty
                except:
                    diff_factor[feature] = (distance / N)


            else:
                print('There has been an exception with:{}'.format(feature))

        else:
            print('There has been an exception with:{}'.format(feature))
    diff_factor['num_particle_diff'] = abs(max(calc["Track_ID"]) - max(compare["Track_ID"]))
    # creating the bar plot
    plt.figure(figsize=(18, 15))
    plt.text(0, 0.5, "TOT_NUM_DIFF_PARTICLES={}".format(diff_factor['num_particle_diff']), style='italic')
    plt.bar(list(diff_factor.keys()), list(diff_factor.values()), width=0.8)
    plt.ylim((0, 1))
    plt.title("{} vs {}".format(prefix, compare_to))
    plt.xticks(list(diff_factor.keys()), rotation='vertical')
    plt.xlabel("Features", fontsize=25)
    plt.ylabel("Difference Factor", fontsize=25)
    plt.tight_layout()
    plt.savefig("Difference Factors of Features {}".format(name_file))
    plt.close()

    return diff_factor


def make_labels(ax, boxplot):
    """
    Prints the statistical values  in the boxplot and returns dictionary of them.
    Parameters
    ----------
    :param ax: Figure in which we are writing to
    :param boxplot: The boxplot of our data

    Returns
    -------
    :return: a dictionary of realevant statistical values iqr, caps, median, outliers, and mean
    """
    # Grab the relevant Line2D instances from the boxplot dictionary
    iqr = boxplot['boxes'][0]
    caps = boxplot['caps']
    med = boxplot['medians'][0]
    fly = boxplot['fliers'][0]
    mean = boxplot['means'][0]

    # The x position of the median line
    xpos = med.get_xdata()

    # horizontal offset which is some fraction of the width of the box
    xoff = 0.10 * (xpos[1] - xpos[0])

    # The x position of the labels
    xlabel = xpos[1] + xoff

    # The median is the y-position of the median line
    median = med.get_ydata()[1]

    # The mean y potisiton
    mean_ = mean.get_ydata()[0]

    # The 25th and 75th percentiles are found from the
    # top and bottom (max and min) of the box
    pc25 = iqr.get_ydata().min()
    pc75 = iqr.get_ydata().max()

    # The caps give the vertical position of the ends of the whiskers
    capbottom = caps[0].get_ydata()[0]
    captop = caps[1].get_ydata()[0]

    # Make some labels on the figure using the values derived above
    ax.text(xlabel, median,
            'Median = {:6.3g}'.format(median), va='center')
    ax.text(xlabel, pc25,
            '25th percentile = {:6.3g}'.format(pc25), va='center')
    ax.text(xlabel + 0.15, mean_,
            'mean = {:6.3g}'.format(mean_), va='center')
    ax.text(xlabel, pc75,
            '75th percentile = {:6.3g}'.format(pc75), va='center')
    ax.text(xlabel, capbottom,
            'Bottom cap = {:6.3g}'.format(capbottom), va='center')
    ax.text(xlabel, captop,
            'Top cap = {:6.3g}'.format(captop), va='center')

    # Many fliers, so we loop over them and create a label for each one
    for outlier in fly.get_ydata():
        ax.text(1 + xoff, outlier,
                'Outlier = {:6.3g}'.format(outlier), va='center')
    return {"median": median, "pc25": pc25, "pc75": pc75, "mean": mean_, "capbottom": capbottom, "captop": captop}


def boxplot_feature(prefix, col_list=["alpha"], outliers=True, file="NaN", umppx=1, fps=1, time=1):
    """
    Plots the Box and Wiskers plot for the list of features given. It prints all relevant values: median, mean, iqrs,
    caps, std and outliers.

    Parameters
    -----------
    :param outliers:
    :param prefix: The dataset name that we are analysing
    :param col_list: The feature names that we want to analyze

    Returns
    -----------
    :return: a dictionary of dictionaries with all the relevant statistical values of the features
    """

    if file == "BM":
        print("BM Diffusion Coeffcient Calculated")
        df = pd.read_csv('{}/DiffCoeffBM_{}.csv'.format(prefix, prefix))
        print(np.mean(df["DiffCoeffBM"]))
    print(file == "D")
    if file == "D":
        print("Diffusion Coeffcient Calculated")
        df = pd.read_csv('{}/features_{}.csv'.format(prefix, prefix), usecols=col_list)  # Read the csv File
        features = list(df.columns)
        for feature in features:
            df[feature] = umppx * df[feature]
    if file == "M":
        print("Mesh Size")
        df = pd.read_csv('{}/Mesh_{}.csv'.format(prefix, prefix))

    if file == "SV":
        print("Speed and Velocity")
        tf = pd.read_csv('{}/Speed.csv'.format(prefix), usecols=col_list)
        tf = tf.iloc[3:].astype('float64')
        df = pd.DataFrame()

        df["TRACK_DISPLACEMENT (µm)"] = umppx * tf["TRACK_DISPLACEMENT"]
        df[" TOTAL_DISTANCE_TRAVELED (µm)"] = umppx * tf["TOTAL_DISTANCE_TRAVELED"]

        if "TRACK_MEAN_SPEED" in col_list:
            df["TRACK_MEAN_SPEED (µm/sec)"] = umppx * tf["TRACK_MEAN_SPEED"] * fps

    if file == "NaN":
        print('Feature File ')
        df = pd.read_csv('{}/features_{}.csv'.format(prefix, prefix), usecols=col_list)  # Read the csv File

    df.dropna(inplace=True)  # Remove all NaN values

    def quantiles(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return Q1, Q3, IQR

    if not outliers:
        check = True
        while check:
            Q1, Q3, IQR = quantiles(df)
            cap_top = df.max(numeric_only=True) > (IQR * 1.5 + Q3) + 1
            cap_bottom = df.min(numeric_only=True) < (Q1 - IQR * 1.5) - 1
            if cap_top.any():
                cap_top = True
            else:
                cap_top = False
            if cap_bottom.any():
                cap_bottom = True
            else:
                cap_bottom = False

            check = cap_top or cap_bottom
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    stds = df.std(axis=0)  # Get a list of  standard deviations for each feature
    df.dropna(inplace=True)  # Remove all NaN values
    # Plot set up
    red_circle = dict(markerfacecolor='red', marker='o')  # Make outliers red circles
    mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='green')  # Plots Mean as a green square

    # Dictionary of Dictionaries for the features
    stat_features = {}
    if len(col_list) > 1:
        fig, axs = plt.subplots(1, len(df.columns), figsize=(20, 10))  # Create the Figure

        for i, ax in enumerate(axs.flat):  # For each of the features
            feat_plot = ax.boxplot(df[df.columns[i]], flierprops=red_circle, showmeans=True, meanprops=mean_shape)
            xpos = feat_plot['medians'][0].get_xdata()  # The x position of the median line
            xoff = 0.10 * (xpos[1] - xpos[0])  # horizontal offset which is some fraction of the width of the box
            xlabel = xpos[1] + xoff - 0.3  # The x position of the std
            text = 'σ={:.2f}'.format(stds[i])  # Standard Deviation
            ax.annotate(text, xy=(xlabel, feat_plot['means'][0].get_ydata()))  # Write the std value
            ax.set_title(df.columns[i], fontsize=20, fontweight='bold')  # Title of the boxplot
            ax.tick_params(axis='y', labelsize=14)
            labels = make_labels(ax, feat_plot)  # Dictionary of stat vals and write them in plot
            labels['std'] = stds[i]  # Add std val to dictionary
            stat_features[df.columns[i]] = labels  # Append dictionary to corresponding feature
    else:  # if there is only one feature that we want to plot
        fig, ax = plt.subplots(1, len(df.columns), figsize=(20, 10))  # Create the Figure
        feat_plot = ax.boxplot(df, flierprops=red_circle, showmeans=True, meanprops=mean_shape)
        xpos = feat_plot['medians'][0].get_xdata()  # The x position of the median line
        xoff = 0.10 * (xpos[1] - xpos[0])  # horizontal offset which is some fraction of the width of the box
        xlabel = xpos[1] + xoff - 0.3  # The x position of the std
        text = 'σ={:.2f}'.format(stds[0])  # Standard Deviation
        ax.annotate(text, xy=(xlabel, feat_plot['means'][0].get_ydata()))  # Write the std value
        ax.set_title(df.columns[0], fontsize=20, fontweight='bold')  # Title of the boxplot
        ax.tick_params(axis='y', labelsize=14)
        labels = make_labels(ax, feat_plot)  # Dictionary of stat vals and write them in plot
        labels['std'] = stds[0]  # Add std val to dictionary
        stat_features[df.columns[0]] = labels  # Append dictionary to corresponding feature

    plt.tight_layout()
    plt.savefig("{}/Box&Wiskers_of_".format(prefix) + '_'.join(col_list))
    return stat_features


def compile_figures(prefix, row_size=10):
    """
    Create a collage of all the plots given by a particular function

    Parameters
    ----------
    :param prefix: String
        Name of function's plots that we want to create a collage from
    :param row_size: Int
        Number of plots in the horizontal

    Returns
    --------
    :return: Plot
        Saves PNG of the collage of plots
    """

    margin = 3  # Distance between plots
    filenames = glob.glob('*{}*'.format(prefix))  # Get all the plot's names
    images = [Image.open(filename) for filename in filenames]  # Open images

    # determines how wide and tall the png will have to be
    width = max(image.size[0] + margin for image in images) * row_size
    height = sum(image.size[1] + margin for image in images)

    montage = Image.new(mode='RGBA', size=(width, height), color=(0, 0, 0, 0))  # Create a new empty png for collage

    # sets where the new image will be added starting from (0,0)
    max_x = 0
    max_y = 0
    offset_x = 0
    offset_y = 0

    # Adds a new image to the montage pngs with the specifications that we have previously set
    for i, image in enumerate(images):
        montage.paste(image, (offset_x, offset_y))

        max_x = max(max_x, offset_x + image.size[0])
        max_y = max(max_y, offset_y + image.size[1])

        if i % row_size == row_size - 1:
            offset_y = max_y + margin
            offset_x = 0
        else:
            offset_x += margin + image.size[0]

    montage = montage.crop((0, 0, max_x, max_y))
    montage.save("{}_Image_Compilation.png".format(prefix))


def particles_in_frame(prefix, x_range=600, y_range=2000):
    """
    Plot number of particles per frame as a function of time.

    Parameters
    ----------
    prefix: string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    x_range: float64 or int
        Desire x range of graph.
    y_range: float64 or int
        Desire y range of graph.
    upload: boolean
        True if you want to upload to s3.

    """
    merged = pd.read_csv('msd_{}.csv'.format(prefix))
    frames = int(max(merged['Frame']))
    framespace = np.linspace(0, frames, frames)
    particles = np.zeros((framespace.shape[0]))
    for i in range(0, frames):
        particles[i] = merged.loc[merged.Frame == i, 'X'].dropna().shape[0]

    fig = plt.figure(figsize=(5, 5))
    plt.plot(framespace, particles, linewidth=4)
    plt.xlim(0, x_range)
    plt.ylim(0, y_range)
    plt.xlabel('Frames', fontsize=20)
    plt.ylabel('Particles', fontsize=20)

    outfile = 'in_frame_{}.png'.format(prefix)
    fig.savefig(outfile, bbox_inches='tight')


def main():
    prefix = "HUVEC-FbChipRED-PEG-PNPS2umFULLAUTO"
    os.chdir(
        '/Users/claudialozano/Dropbox/PycharmProjects/AD_nanoparticle/diff_classifier/notebooks/development/MPT_Data/')
    boxplot_feature(prefix, col_list=["DiffCoeffBM"], file="BM", umppx=0.8014, fps=5, time = 0.2)
    # plot_individual_features(prefix, col_list=["Track_ID","Mean alpha", "Mean Deff1", "Mean D_fit","alpha","MSD_ratio"])
    os.chdir(
        '/Users/claudialozano/Dropbox/PycharmProjects/AD_nanoparticle/diff_classifier/notebooks/development/MPT_Data/{}'.format(
            prefix))
    # particles_in_frame(prefix, x_range=500, y_range=50)

    #plot_compare_features("Matrigel_TrackMate_Full", "Matrigel_TrackMate_Full_mine_more",
                          #"TrackMate Consistency Quality auto")

    # plot_histogram(prefix, xlabel='Diffusion Coefficient Dist', ylabel='Trajectory Count',
    #                fps=5, umppx= 0.8014, frames=119, y_range=40, frame_interval=10, frame_range=50,
    #                analysis='NaN', theta='D')


if __name__ == "__main__":
    main()
