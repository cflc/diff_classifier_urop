"""
Functions to calculate mean squared displacements from trajectory data .This module includes functions to calculate
mean squared displacements and additional measures from input trajectory datasets as calculated by theTrackmate ImageJ
plugin.

This file has been edited by Claudia Lozano for readability and familiarization of the code. The original can be found
in the original_msd.py file.


"""
import warnings
import random as rand
import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import diff_classifier.aws as aws
from scipy.ndimage.morphology import distance_transform_edt as eudist


def nth_diff(dataframe, n=1, axis=0):
    """
    Calculates the nth difference between vector elements
            The nth difference is the difference between a number x and a number y n places away

    Returns a new vector of size N - n containing the nth difference between vector elements.

    Parameters
    ----------
    :param dataframe: pandas.core.series.Series of int or float
        Input data on which differences are to be calculated.
    :param n:  int
        Function calculated xpos(i) - xpos(i - n) for all values in pandas
        series. Automatically set to 1
    :param axis: {0, 1}
        Axis along which differences are to be calculated.  Default is 0.  If 0,
        input must be a pandas series.  If 1, input must be a numpy array.

    Returns
    -------
    :return: diff : pandas.core.series.Series of int or float
        Pandas series of size N - n, where N is the original size of dataframe.

    Examples
    --------
    >>> df = np.ones((5, 10))
    >>> nth_diff(df)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    >>> df = pd.DataFrame( [1,2,3,4,5,6,7])
    >>> nth_diff(df, n=1)
    array([ 1,1,1,1,1,1)]
    >>> nth_diff(df, n=2)
    array([ 2,2,2,2,2])
    """

    assert isinstance(n, int), "n must be an integer."  # assert n is an integer
    length = dataframe.shape[0]
    print("This function takes the parameters: DataFrame, nth value and axis")
    if dataframe.ndim == 1:
        if n <= length:
            test1 = dataframe[:-n].reset_index(drop=True)  # return df from 0 to length - n
            test2 = dataframe[n:].reset_index(drop=True)  # return from the nth to the last item
            diff = test2 - test1
        else:
            # if n is greater than the length we cannot compare, return NaN
            diff = np.array([np.nan, np.nan])

    else:  # if more than 1 dimension
        if n <= length:
            #   using the same logic as above in the appropriate axis
            if axis == 0:
                test1 = dataframe[:-n, :]
                test2 = dataframe[n:, :]
            else:
                test1 = dataframe[:, :-n]
                test2 = dataframe[:, n:]
            diff = test2 - test1

        else:
            # if n is greater than the length we cannot compare, return NaN
            diff = np.array([np.nan, np.nan])

    return diff


def make_xyarray(data):
    """
    Rearranges xy position data into 2d arrays
    Rearranges xy data from input pandas dataframe into 2D numpy array.

    This takes the particles from the different track and then combines it so that the x positions
    at the same time are put in together

    example:
    - if we have 2 particles a and b and at frame 1 a is at position X =3 and  b is not in frame, Then we have [3, nan]
    - if on frame 2 a postion X=7 and b position X= 8 then we have [[3,nan],[7,8]]

    Parameters
    ----------
    :param data : pd.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'Track_ID', 'X', and 'Y' column.
    :param length : int
        Desired length or number of frames to which to extend trajectories.
        Any trajectories shorter than the input length will have the extra space
        filled in with NaNs.

    Returns
    -------
    :return xyft : dict of np.ndarray
        Dictionary containing xy position data, frame data, and trajectory ID data. Contains the following keys:
        - farray, frames data (length x particles)
        - tarray, trajectory ID data (length x particles)
        - xarray, x position data (length x particles)
        - yarray, y position data (length x particles)

    Examples
    --------
    >>>  data1 = {'Frame': [0, 1, 2, 3, 4, 2, 3, 4, 5, 6], 'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    >>>'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6], 'Quality': [5,5,4,6,7,5,4,4,3,5],
    >>> 'Mean_Intensity': [100,300,200,300,300,300,400,300,400,300],'SN_Ratio':[0.9,0.8,0.9,1,1,0.9,1,0.8,0.9,1]}
    >>> df = pd.DataFrame(data=data1)
    >>> length = max(df['Frame']) + 1
    >>> xyft = make_xyarray(df, length=length)
    {'farray': array([[0., 0.],[1., 1.],[2., 2.], [3., 3.], [4., 4.], [5., 5.], [6., 6.]]),
     'tarray': array([[1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]),
     'xarray': array([[ 5., nan], [ 6., nan], [ 7.,  1.], [ 8.,  2.], [ 9.,  3.], [nan,  4.],
     'yarray': [nan,  5.]]),array([[ 6., nan],[ 7., nan],[ 8.,  2.],[ 9.,  3.],[10.,  4.],[nan,  5.],[nan,  6.]])}
    """
    print("The only parameter for this function is the data with Frame, Track_ID, X, and Y arrays ")
    length = int(max(data['Frame']) + 1)
    # Initial values
    first_p = int(min(data['Track_ID']))  # this is the smallest Track ID value (1)
    particles = int(max(data['Track_ID'])) - first_p + 1  # number of IDs , ie. total number of particles
    new_frame = np.linspace(0, length - 1, length)

    xyft = {'xarray': np.zeros((length, particles)), 'yarray': np.zeros((length, particles)),
            'farray': np.zeros((length, particles)), 'tarray': np.zeros((length, particles)),
            'qarray': np.zeros((length, particles)), 'snarray': np.zeros((length, particles)),
            'iarray': np.zeros((length, particles))}

    # calculate the msd values for each fo the particles
    for part in range(first_p, first_p + particles):
        track = data[data['Track_ID'] == part].sort_values(['Track_ID', 'Frame'], ascending=[1, 1]).reset_index(
            drop=True)

        old_frame = track['Frame']
        oldxy = [track['X'].values,
                 track['Y'].values,
                 track['Quality'].values,
                 track['SN_Ratio'].values,
                 track['Mean_Intensity'].values]

        # interpolate the values of the old frame (smooth out between points)
        fxy = [interpolate.interp1d(old_frame, oldxy[0], bounds_error=False, fill_value=np.nan),
               interpolate.interp1d(old_frame, oldxy[1], bounds_error=False, fill_value=np.nan),
               interpolate.interp1d(old_frame, oldxy[2], bounds_error=False, fill_value=np.nan),
               interpolate.interp1d(old_frame, oldxy[3], bounds_error=False, fill_value=np.nan),
               interpolate.interp1d(old_frame, oldxy[4], bounds_error=False, fill_value=np.nan)]

        intxy = [fxy[0](new_frame), fxy[1](new_frame), fxy[2](new_frame), fxy[3](new_frame), fxy[4](new_frame)]

        # fill out array with the interpolated values for each particle (part - first_p saves the data to the different
        # particles offset by first_p, which is usually 1)
        xyft['xarray'][:, part - first_p] = intxy[0]
        xyft['yarray'][:, part - first_p] = intxy[1]
        xyft['farray'][:, part - first_p] = new_frame
        xyft['tarray'][:, part - first_p] = part
        xyft['qarray'][:, part - first_p] = intxy[2]
        xyft['snarray'][:, part - first_p] = intxy[3]
        xyft['iarray'][:, part - first_p] = intxy[4]

    return xyft


def msd_calc(track, length=10):
    """Calculates mean squared displacement of input track.
    Returns numpy array containing MSD data calculated from an individual track.

    Parameters
    ----------
    :param track : pandas.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'X', and 'Y' column"

    :param length : The maximum nth difference calculated i.e. the total number of frames

    Returns
    -------
    :return new_track : pandas.core.frame.DataFrame
        - Similar to input track.
        - All missing frames of individual trajectories are filled in with NaNs
        - two new columns, MSDs and Gauss are added
        - units are in px^2

        MSDs, calculated mean squared displacements using the formula: MSD = <(xpos-x0)**2>
        Gauss, calculated Gaussianity (The extent to which something is Gaussian)

    Examples
    --------
    >>> data1 = {'Frame': [1, 2, 3, 4, 5],
    ...          'X': [5, 6, 7, 8, 9],
    ...          'Y': [6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data=data1)
    >>> new_track = msd.msd_calc(df, 5)

    """

    meansd = np.zeros(length)
    gauss = np.zeros(length)
    new_frame = np.linspace(1, length, length)
    old_frame = track['Frame']
    oldxy = [track['X'], track['Y']]

    # x and y positions with interpolated calculated values
    fxy = [interpolate.interp1d(old_frame, oldxy[0], bounds_error=False,
                                fill_value=np.nan),
           interpolate.interp1d(old_frame, oldxy[1], bounds_error=False,
                                fill_value=np.nan)]
    """
    PREVIOUS ATTEMPT THIS DOES NOT WORK BECAUSE NAN IS NOT EQUAL TO NAN
    intxy = [ma.masked_equal(fxy[0](new_frame), np.nan),
             ma.masked_equal(fxy[1](new_frame), np.nan)]
    """
    intxy = [ma.masked_invalid(fxy[0](new_frame)), ma.masked_invalid(fxy[1](new_frame))]  # masks the NaN values

    data1 = {'Frame': new_frame, 'X': intxy[0], 'Y': intxy[1]}
    new_track = pd.DataFrame(data=data1)

    for frame in range(1, length):
        # square the nth differences where n increases each time
        xy = [np.square(nth_diff(new_track['X'], n=frame)),
              np.square(nth_diff(new_track['Y'], n=frame))]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meansd[frame] = np.nanmean(xy[0] + xy[1])
            # calculated Gaussianity (The extent to which something is Gaussian)
            gauss[frame] = np.nanmean(xy[0] ** 2 + xy[1] ** 2) / (2 * (meansd[frame] ** 2))

    new_track['MSDs'] = pd.Series(meansd, index=new_track.index)
    new_track['Gauss'] = pd.Series(gauss, index=new_track.index)

    return new_track


def all_msds2(data):
    """
    Calculates mean squared displacements of input trajectory dataset
    Returns numpy array containing MSD data of all tracks in a trajectory pandas
    dataframe.

    Parameters
    ----------
    :param data : pandas.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'Track_ID', 'X', and
        'Y' column. Note: it is assumed that frames begins at 0.

    Returns
    -------
    :return new_data : pandas.core.frame.DataFrame
        - Similar to input track.
        - All missing frames of individual trajectories are filled in with NaNs
        - two new columns, MSDs and Gauss are added
        - units are in px^2

        MSDs, calculated mean squared displacements using the formula: MSD = <(xpos-x0)**2>
        Gauss, calculated Gaussianity (The extent to which something is Gaussian)

    Examples
    --------
    >>> data1 = {'Frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    ...          'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    ...          'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
    ...          'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    >>> df = pd.DataFrame(data=data1)
    >>> cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss']
    >>> length = max(df['Frame']) + 1
    >>> all_msds2(df, frames=length)[cols]
    """
    print("The only parameter for this function is the data with Frame, Track_ID, X, and Y arrays ")
    if data.shape[0] > 2:  # at least 2 Frames/IDs/X-val s/Y-vals
        try:
            frames = max(data['Frame']) + 1  # The way they calculate frame value in the MSD TEST file
            xyft = make_xyarray(data)
            length = xyft['xarray'].shape[0]  # number of frames
            particles = xyft['xarray'].shape[1]  # number of TrackID

            meansd = np.zeros((length, particles))
            gauss = np.zeros((length, particles))

            for frame in range(1, length):
                # For each of the frames except the last calculate the nth difference where n is the frame whe are in
                # + 1. the nth difference array is then squared per the MSD equation
                xpos = np.square(nth_diff(xyft['xarray'], n=frame))  # (xpos-x0)**2
                ypos = np.square(nth_diff(xyft['yarray'], n=frame))  # (ypos-x0)**2

                # TODO: Check if the way to do 2D MSD is by adding the x MSD and the Y MSD
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    meansd[frame, :] = np.nanmean(xpos + ypos, axis=0)  # We add the mean value of all the MSDs values
                    gauss[frame, :] = np.nanmean(xpos ** 2 + ypos ** 2, axis=0) / (2 * (meansd[frame] ** 2))

            # TODO: Check what those np.flatten("F") Do.
            data1 = {'Frame': xyft['farray'].flatten('F'),
                     'Track_ID': xyft['tarray'].flatten('F'),
                     'X': xyft['xarray'].flatten('F'),
                     'Y': xyft['yarray'].flatten('F'),
                     'MSDs': meansd.flatten('F'),
                     'Gauss': gauss.flatten('F'),
                     'Quality': xyft['qarray'].flatten('F'),
                     'SN_Ratio': xyft['snarray'].flatten('F'),
                     'Mean_Intensity': xyft['iarray'].flatten('F')}

            new_data = pd.DataFrame(data=data1)

        # IN CASE OF AN ERROR RETURN A DF WITH EMPTY LISTS
        except ValueError:
            print('YOU DID NOT GET AN MSD2 VALUE ERROR')
            data1 = {'Frame': [], 'Track_ID': [], 'X': [], 'Y': [], 'MSDs': [], 'Gauss': [],
                     'Quality': [], 'SN_Ratio': [], 'Mean_Intensity': []}
            new_data = pd.DataFrame(data=data1)
        except IndexError:
            print('YOU DID NOT GET AN MSD2 INDEX ERROR')
            data1 = {'Frame': [], 'Track_ID': [], 'X': [], 'Y': [], 'MSDs': [], 'Gauss': [],
                     'Quality': [], 'SN_Ratio': [], 'Mean_Intensity': []}
            new_data = pd.DataFrame(data=data1)
    else:
        print('YOU DID NOT GET AN MSD2 ELSE')
        data1 = {'Frame': [], 'Track_ID': [], 'X': [], 'Y': [], 'MSDs': [], 'Gauss': [], 'Quality': [],
                 'SN_Ratio': [], 'Mean_Intensity': []}
        new_data = pd.DataFrame(data=data1)

    return new_data


# ASSUMES UMPPX, FPS, BACK UP FRAMES
def geomean_msdisp(prefix, umppx=0.16, backup_frames=651):
    """
    Computes geometric averages of mean squared displacement datasets
    Calculates geometric averages and stnadard errors for MSD datasets. Might
    error out if not formatted as output from all_msds2.

    Parameters
    ----------
    :param prefix : string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    :param umppx : float
        Microns per pixel of original images.
    :param backup_frames : Number of frames in the video

    Returns
    -------
    :return geo_mean : numpy.ndarray
        Geometric mean of trajectory MSDs at all time points.
    :return geo_stder : numpy.ndarray
        Geometric standard error of trajectory MSDs at all time points.

    """
    print("The parameters for this function are prefix name, resolution in umppx and the backup frame number")
    merged = pd.read_csv('msd_{}.csv'.format(prefix))
    try:
        particles = int(max(merged['Track_ID']))  # number of particles
        frames = int(max(merged['Frame']))  # number of frames
        msd_val = np.zeros((particles + 1, frames + 1))  # initiate the MSD vals array

        for i in range(0, particles + 1):
            # get the corresponding MSD value for each of the particles
            # NOTE: it has already taken into account the resolution
            msd_val[i, :] = merged.loc[merged.Track_ID == i, 'MSDs'] * umppx * umppx

        # TODO: What is ma.log. What is a geometic mean? What is ma.masked_equal? WHAT IS STATS.SEM?
        geo_mean = np.nanmean(ma.log(msd_val), axis=0)
        geo_stder = ma.masked_equal(stats.sem(ma.log(msd_val), axis=0, nan_policy='omit'), 0.0)

    except ValueError:
        print("NO GEOMEAN OR GEOSTD CALC. VLAUE ERROR ")
        geo_mean = np.nan * np.ones(backup_frames)
        geo_stder = np.nan * np.ones(backup_frames)

    np.savetxt('geomean_{}.csv'.format(prefix), geo_mean, delimiter=",")
    np.savetxt('geoSEM_{}.csv'.format(prefix), geo_stder, delimiter=",")

    return geo_mean, geo_stder


def binning(experiments, wells=4, prefix='test'):
    """Split set of input experiments into groups.

    Parameters
    ----------
    :param experiments : list of str
        List of experiment names.
    :param wells : int
        Number of groups to divide experiments into.
    :param prefix: str
        Name of the file where data is stored
    Returns
    -------
    slices : int
        Number of experiments per group.
    bins : dict of list of str
        Dictionary, keys corresponding to group names, and elements containing
        lists of experiments in each group.
    bin_names : list of str
        List of group names
    """
    print("The parameters in for this functions are number of wells and prefix name")
    total_videos = len(experiments)  # number of experiments
    slices = int(total_videos / wells)  # division of the videos such that we divided in to the well number
    bins = {}  # initialize bin dictionary
    bin_names = []  # name of bins list

    for num in range(0, wells):
        slice1 = num * slices  # beginning of the slices
        slice2 = (num + 1) * slices  # the end of the experiments in the slice
        pref = '{}_W{}'.format(prefix, num)
        bins[pref] = experiments[slice1:slice2]
        bin_names.append(pref)

    # return the num of exp per group, the groups corresponding to each group name, and a list of group names
    return slices, bins, bin_names


def precision_weight(group, geo_stder):
    """
    Calculates precision-based weights from input standard error data
    Calculates precision weights to be used in precision-averaged MSD calculations.

    Parameters
    ----------
    :param group : list of str
        List of experiment names to average. Each element corresponds to a key in geo_stder and geomean.
    :param geo_stder : dict of numpy.ndarray
        Each entry in dictionary corresponds to the standard errors of an MSD profile, the key corresponding to an
        experiment name.
            - geostder of the geomean_msdisp function

    Returns
    -------
    :return weights: numpy.ndarray
        Precision weights to be used in precision averaging.
    :return w_holder : numpy.ndarray
        Precision values of each video at each time point.

    """

    w_holder = []
    for sample in group:
        w_holder.append(1 / (geo_stder[sample] * geo_stder[sample]))  # Calculate the weight of each sample

    # Get rid of the 0 and 1's values since they are not unseful
    w_holder = ma.masked_equal(w_holder, 0.0)
    w_holder = ma.masked_equal(w_holder, 1.0)

    weights = ma.sum(w_holder, axis=0)  # the final weight is the sum of the weights of each group's item weight

    return weights, w_holder


def precision_averaging(group, geomean, geo_stder, weights):
    """Calculates precision-weighted averages of MSD datasets.

    Parameters
    ----------
    group : list of str
        List of experiment names to average. Each element corresponds to a key
        in geo_stder and geomean.
    geomean : dict of numpy.ndarray
        Each entry in dictionary corresponds to an MSD profiles, they key
        corresponding to an experiment name.
    geo_stder : dict of numpy.ndarray
        Each entry in dictionary corresponds to the standard errors of an MSD
        profile, the key corresponding to an experiment name.
    weights : numpy.ndarray
        Precision weights to be used in precision averaging.

    Returns
    -------
    geo : numpy.ndarray
        Precision-weighted averaged MSDs from experiments specified in group
    geo_stder : numpy.ndarray
        Precision-weighted averaged SEMs from experiments specified in group
        :param weights:
        :param geo_stder:
        :param group:
        :param geomean:
    """

    frames = np.shape(geo_stder[group[0]])[0]
    slices = len(group)

    video_counter = 0
    geo_holder = np.zeros((slices, frames))
    gstder_holder = np.zeros((slices, frames))
    w_holder = np.zeros((slices, frames))
    for sample in group:
        w_holder[video_counter, :] = (1 / (geo_stder[sample] * geo_stder[sample])
                                      ) / weights
        geo_holder[video_counter, :] = w_holder[video_counter, :
                                       ] * geomean[sample]
        gstder_holder[video_counter, :] = 1 / (geo_stder[sample] * geo_stder[sample]
                                               )
        video_counter = video_counter + 1

    w_holder = ma.masked_equal(w_holder, 0.0)
    w_holder = ma.masked_equal(w_holder, 1.0)
    geo_holder = ma.masked_equal(geo_holder, 0.0)
    geo_holder = ma.masked_equal(geo_holder, 1.0)
    gstder_holder = ma.masked_equal(gstder_holder, 0.0)
    gstder_holder = ma.masked_equal(gstder_holder, 1.0)

    geo = ma.sum(geo_holder, axis=0)
    geo_stder = ma.sqrt((1 / ma.sum(gstder_holder, axis=0)))

    geodata = Bunch(geomean=geo, geostd=geo_stder, weighthold=w_holder,
                    geostdhold=gstder_holder)

    return geodata


def plot_all_experiments(experiments, bucket='ccurtis.data', folder='test',
                         yrange=(10 ** -1, 10 ** 1), fps=100.02,
                         xrange=(10 ** -2, 10 ** 0),
                         outfile='test.png', exponential=True,
                         labels=None, log=True):
    """Plots precision-weighted averages of MSD datasets.

    Plots pre-calculated precision-weighted averages of MSD datasets calculated
    from precision_averaging and stored in an AWS S3 bucket.

    Parameters
    ----------
    :param yrange : list of float
        Y range of plot
    :param xrange: list of float
        X range of plot
    :param outfile : str
        Filename of output image
    """

    n = len(experiments)

    if labels is None:
        labels = experiments

    color = iter(cm.viridis(np.linspace(0, 0.9, n)))

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111)
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    plt.xlabel('Tau (s)', fontsize=25)
    plt.ylabel(r'Mean Squared Displacement ($\mu$m$^2$)', fontsize=25)

    geo = {}
    gstder = {}
    counter = 0
    for experiment in experiments:
        aws.download_s3('{}/geomean_{}.csv'.format(folder, experiment),
                        'geomean_{}.csv'.format(experiment), bucket_name=bucket)
        aws.download_s3('{}/geoSEM_{}.csv'.format(folder, experiment),
                        'geoSEM_{}.csv'.format(experiment), bucket_name=bucket)

        geo[counter] = np.genfromtxt('geomean_{}.csv'.format(experiment))
        gstder[counter] = np.genfromtxt('geoSEM_{}.csv'.format(experiment))
        geo[counter] = ma.masked_equal(geo[counter], 0.0)
        gstder[counter] = ma.masked_equal(gstder[counter], 0.0)

        frames = np.shape(gstder[counter])[0]
        xpos = np.linspace(0, frames - 1, frames) / fps
        c = next(color)

        if exponential:
            ax.plot(xpos, np.exp(geo[counter]), c=c, linewidth=6,
                    label=labels[counter])
            ax.fill_between(xpos, np.exp(geo[counter] - 1.96 * gstder[counter]),
                            np.exp(geo[counter] + 1.96 * gstder[counter]),
                            color=c, alpha=0.4)

        else:
            ax.plot(xpos, geo[counter], c=c, linewidth=6,
                    label=labels[counter])
            ax.fill_between(xpos, geo[counter] - 1.96 * gstder[counter],
                            geo[counter] + 1.96 * gstder[counter], color=c,
                            alpha=0.4)

        counter = counter + 1

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.legend(frameon=False, loc=2, prop={'size': 16})
    fig.savefig(outfile, bbox_inches='tight')


def checkerboard_mask(dims=(512, 512), squares=50, width=25):
    """Creates a 2D Boolean checkerboard mask

    Creates a Boolean array of evenly spaced squares.
    Whitespace is set to True.

    Parameters
    ----------

    dims : tuple of int
        Dimensions of desired Boolean array
    squares : int
        Dimensions of in individual square in array
    width : int
        Dimension of spacing between squares

    Returns
    ----------

    zeros : numpy.ndarray of bool
        2D Boolean array of evenly spaced squares

    """
    zeros = np.zeros(dims) == 0
    square_d = squares

    loy = width
    hiy = loy + square_d

    for j in range(50):

        lox = width
        hix = lox + square_d
        for i in range(50):

            if hix < 512 and hiy < 512:
                zeros[loy:hiy, lox:hix] = False
            elif hix < 512:
                zeros[loy:512 - 1, lox:hix] = False
            elif hiy < 512:
                zeros[loy:hiy, lox:512 - 1] = False
            else:
                zeros[loy:512 - 1, lox:512 - 1] = False
                break

            lox = hix + width
            hix = lox + square_d

        loy = hiy + width
        hiy = loy + square_d

    return zeros


def random_walk(nsteps=100, seed=None, start=(0, 0), step=1, mask=None,
                stuckprob=0.5):
    # RANDOM WALK FUNCTION IS NOT WORKING EVEN USING THE ORIGINAL CODE
    # WILL HAVE TO DO FURHTER EDITING
    """Creates 2d random walk trajectory.

    Parameters
    ----------
    nsteps : int
        Number of steps for trajectory to move.
    seed : int
        Seed for pseudo-random number generator for reproducability.
    start : tuple of int or float
        Starting xy coordinates at which the random walk begins.
    step : int or float
        Magnitude of single step
    mask : numpy.ndarray of bool
        Mask of barriers contraining diffusion
    stuckprop : float
        Probability of "particle" adhering to barrier when it makes contact

    Returns
    -------
    x : numpy.ndarray
        Array of x coordinates of random walk.
    y : numpy.ndarray
        Array of y coordinates of random walk.

    """

    if type(mask) is np.ndarray:
        while not mask[start[0], start[1]]:
            start = (start[0], start[1] + 1)
        eumask = eudist(~mask)

    np.random.seed(seed=seed)

    x = np.zeros(nsteps)
    y = np.zeros(nsteps)
    x[0] = start[0]
    y[0] = start[1]

    # Checks to see if a mask is being used first
    if not type(mask) is np.ndarray:
        for i in range(1, nsteps):
            val = rand.randint(1, 4)
            if val == 1:
                x[i] = x[i - 1] + step
                y[i] = y[i - 1]
            elif val == 2:
                x[i] = x[i - 1] - step
                y[i] = y[i - 1]
            elif val == 3:
                x[i] = x[i - 1]
                y[i] = y[i - 1] + step
            else:
                x[i] = x[i - 1]
                y[i] = y[i - 1] - step
    else:
        # print("Applied mask")
        for i in range(1, nsteps):
            val = rand.randint(1, 4)
            # If mask is being used, checks if entry is in mask or not
            if mask[int(x[i - 1]), int(y[i - 1])]:
                if val == 1:
                    x[i] = x[i - 1] + step
                    y[i] = y[i - 1]
                elif val == 2:
                    x[i] = x[i - 1] - step
                    y[i] = y[i - 1]
                elif val == 3:
                    x[i] = x[i - 1]
                    y[i] = y[i - 1] + step
                else:
                    x[i] = x[i - 1]
                    y[i] = y[i - 1] - step
            # If it does cross into a False area, probability to be stuck
            elif np.random.rand() > stuckprob:
                x[i] = x[i - 1]
                y[i] = y[i - 1]

                while eumask[int(x[i]), int(y[i])] > 0:
                    vals = np.zeros(4)
                    vals[0] = eumask[int(x[i] + step), int(y[i])]
                    vals[1] = eumask[int(x[i] - step), int(y[i])]
                    vals[2] = eumask[int(x[i]), int(y[i] + step)]
                    vals[3] = eumask[int(x[i]), int(y[i] - step)]
                    vali = np.argmin(vals)

                    if vali == 0:
                        x[i] = x[i] + step
                        y[i] = y[i]
                    elif vali == 1:
                        x[i] = x[i] - step
                        y[i] = y[i]
                    elif vali == 2:
                        x[i] = x[i]
                        y[i] = y[i] + step
                    else:
                        x[i] = x[i]
                        y[i] = y[i] - step
            # Otherwise, particle is stuck on "cell"
            else:
                x[i] = x[i - 1]
                y[i] = y[i - 1]

    return x, y


def random_traj_dataset(nframes=100, nparts=30, seed=1, fsize=(0, 512),
                        ndist=(1, 2)):
    """
    Creates a random population of random walks.

    Parameters
    ----------
    :param nframes : int
        Number of frames for each random trajectory.
    :param nparts : int
        Number of particles in trajectory dataset.
    :param seed : int
        Seed for pseudo-random number generator for reproducability.
    :param fsize : tuple of int or float
        Scope of points over which particles may start at.
    :param ndist : tuple of int or float
        Parameters to generate normal distribution, mu and sigma.

    Returns
    -------
    dataf : pandas.core.frame.DataFrame
        Trajectory data containing a 'Frame', 'Track_ID', 'X', and
        'Y' column.

    """

    frames = []
    trackid = []
    x = []
    y = []
    start = [0, 0]
    pseed = seed

    for i in range(nparts):
        rand.seed(a=i + pseed)
        start[0] = rand.randint(fsize[0], fsize[1])
        rand.seed(a=i + 3 + pseed)
        start[1] = rand.randint(fsize[0], fsize[1])
        rand.seed(a=i + 5 + pseed)
        weight = rand.normalvariate(mu=ndist[0], sigma=ndist[1])

        trackid = np.append(trackid, np.array([i] * nframes))
        xi, yi = random_walk(nsteps=nframes, seed=i)
        x = np.append(x, weight * xi + start[0])
        y = np.append(y, weight * yi + start[1])
        frames = np.append(frames, np.linspace(0, nframes - 1, nframes))

    datai = {'Frame': frames,
             'Track_ID': trackid,
             'X': x,
             'Y': y,
             'Quality': nframes * nparts * [10],
             'SN_Ratio': nframes * nparts * [0.1],
             'Mean_Intensity': nframes * nparts * [120]}
    dataf = pd.DataFrame(data=datai)

    return dataf


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def main():
    experiments = []
    geomean = {}
    geostder = {}
    for num in range(4):
        name = 'test_{}'.format(num)
        experiments.append(name)
        data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                 'Track_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                 'X': [x * (num + 1) for x in [5, 6, 7, 8, 9, 2, 4, 6, 8, 10]],
                 'Y': [x * (num + 1) for x in [6, 7, 8, 9, 10, 6, 8, 10, 12, 14]],
                 'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                 'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data=data1)
        msds = all_msds2(df)
        msds.to_csv('msd_test_{}.csv'.format(num))
        geomean[name], geostder[name] = geomean_msdisp(name, umppx=1)
    weights, w_holder = precision_weight(experiments, geostder)
    print("weights", weights)
    print("w_holder", w_holder)


if __name__ == "__main__":
    main()
"""
weights [-- 8.325475924022431 8.32547592402243 8.32547592402243 8.32547592402243
 --]
w_holder [[-- 2.0813689810056086 2.0813689810056073 2.0813689810056073
  2.0813689810056086 --]
 [-- 2.0813689810056073 2.0813689810056086 2.0813689810056073
  2.0813689810056073 --]
 [-- 2.0813689810056073 2.0813689810056073 2.0813689810056073
  2.0813689810056073 --]
 [-- 2.0813689810056086 2.0813689810056073 2.0813689810056073
  2.0813689810056073 --]]"""
