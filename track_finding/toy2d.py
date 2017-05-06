"""
This module contains helper code for generating and manipulating toy detector
data for the ML algorithms.

So far it just has the 1D detector straight-track data generation.
"""
from __future__ import print_function

import numpy as np

def calc_mask_ranges(det_width, mask_shapes):
    """
    Calculate the indices of the detector mask envelope.
    Parameters:
        det_width: width of the 1D detector
        mask_shapes: ndarray of widths of the mask
    Returns:
        Two arrays representing the lower and upper index ranges of the detector mask.
    """
    lower = ((det_width - mask_shapes) / 2).astype(int)
    upper = lower + mask_shapes
    return lower, upper

def get_mask_ranges(det_shape, mask_shapes):
    """
    Calculate the indices of the detector mask envelope.
    Meant to replace the above function.
    Parameters:
        det_shape: shape of the full 1D detector, (depth, width)
        mask_shapes: ndarray of widths of the mask
    Returns:
        ndarray of lower, upper indices in each 1D layer; shape=(depth, 2)
    """
    det_depth, det_width = det_shape
    assert det_depth == mask_shapes.shape[0]
    ranges = np.zeros((det_depth, 2), int)
    ranges[:,0] = (det_width - mask_shapes) / 2
    ranges[:,1] = ranges[:,0] + mask_shapes
    return ranges

def construct_mask(det_shape, mask_shapes):
    """
    Construct the boolean mask used to select a wedge of the detector.
    Parameters:
        det_shape: shape of the full 1D detector
        mask_shapes: ndarray of widths of the mask
    Returns:
        Boolean array of the detector mask, with dimensions matching det_shape.
    """
    det_mask = np.zeros(det_shape, bool)
    mask_ranges = get_mask_ranges(det_shape, mask_shapes)
    for i, r in enumerate(mask_ranges):
        det_mask[i, r[0]:r[1]] = True
    return det_mask

def apply_det_mask(data, mask):
    """
    Apply detector mask to 1D detector data events.
    Parameters:
        data: ndarray of 1D detector events
        mask: boolean detector mask ndarray
    Returns:
        List of masked layer data arrays.
    """
    assert data[0].shape == mask.shape, \
        'shapes unequal: {} != {}'.format(data[0].shape, mask.shape)
    # Group event data by masked layers
    return [data[:,ilayer,mask[ilayer]] for ilayer in range(mask.shape[0])]

def expand_masked_data(masked_data, mask):
    """
    Unmask detector data and expand into fixed-size detector array.
    Parameters:
        masked_data: list of ndarrays of detector layer data
            for multiple events
        mask: boolean detector mask used to mask the data
    Returns:
        ndarray of data where each event is same shape as the mask
    """
    # Let's first assume that all layers are present.
    # I will still need to handle the case where first or last layer is dropped.
    assert len(masked_data) == mask.shape[0], \
        'Data shape incompatible with detector mask'
    output_shape = (len(masked_data[0]),) + mask.shape
    output = np.zeros(output_shape)
    # Loop over layers
    for ilayer, mask in enumerate(mask):
        output[:,ilayer,mask] = masked_data[ilayer]
    return output

def simulate_straight_track(m, b, det_shape):
    """
    Simulate detector data for one straight track.
    Parameters:
        m: track slope parameter
        b: track first-layer intercept parameter (detector entry point)
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of binary detector data for one track.
    """
    x = np.zeros(det_shape)
    idx = np.arange(det_shape[0])
    hits = (idx*m + b).astype(int)
    x[idx, hits] = 1
    return x

def generate_straight_track(det_shape):
    """
    Sample track parameters and simulate detector data.
    Parameters:
        det_shape: tuple of detector shape: (depth, width)
    Returns:
        ndarray of binary detector data for one track.
    """
    det_depth, det_width = det_shape
    # Sample detector entry point
    b = np.random.random_sample()*(det_width - 1)
    # Sample detector exit point
    b2 = np.random.random_sample()*(det_width - 1)
    # Calculate track slope
    m = (b2 - b) / det_depth
    return simulate_straight_track(m, b, det_shape)

def generate_straight_tracks(n, det_shape, entry_range=None, exit_range=None):
    """
    Generates single straight-track events in 1D detector.
    Parameters:
        n: number of single-track events to generate
        det_shape: tuple of detector shape: (depth, width)
        entry_range: range tuple to sample detector entry point
        exit_range: range tuple to sample detector exit point
    Returns:
        ndarray of detector data for n single-track events. The shape is
        (n, det_shape[0], det_shape[1]).
    """
    # Initialize the data
    det_depth, det_width = det_shape
    data = np.zeros((n, det_depth, det_width))
    # Sample detector entry and exit points
    if entry_range is None: entry_range = (0, det_width)
    if exit_range is None: exit_range = (0, det_width)
    entry_points = np.random.uniform(size=n, *entry_range)
    exit_points = np.random.uniform(size=n, *exit_range)
    # Calculate track slopes
    slopes = (exit_points - entry_points) / det_depth
    # Simulate detector response and fill the data structure
    for i, (entry, slope) in enumerate(zip(entry_points, slopes)):
        data[i] = simulate_straight_track(slope, entry, det_shape)
    return data

def generate_uniform_noise(n, det_shape, prob=0.1, skip_layers=5):
    """
    Generate uniform noise hit data.
    Parameters:
        n: number of noise events to generate
        det_shape: tuple of detector shape: (depth, width)
        prob: probability of noise hit in each pixel
        skip_layers: number of detector layers to skip (no noise)
    Returns:
        ndarray of detector noise data for n events. The shape is
        (n, det_shape[0], det_shape[1]).
    """
    # One way to do this: generate random floats in [0,1]
    # and then convert the ones above threshold to binary
    det_depth, det_width = det_shape
    noise_events = np.zeros([n, det_depth, det_width])
    rand_vals = np.random.random_sample([n, det_depth-skip_layers, det_width])
    noise_events[:,skip_layers:,:] = rand_vals < prob
    return noise_events

def generate_track_bkg(n, det_shape, tracks_per_event=2, skip_layers=5):
    """
    Generate events with a number of clean layers followed by layers containing
    background tracks.

    Parameters:
        n: number of events to generate
        det_shape: tuple of detector shape: (depth, width)
        tracks_per_event: fixed number of tracks to simulate in each event
        skip_layers: number of detector layers to skip (no bkg)
    Returns:
        ndarray of detector data for n events. The shape is
        (n, det_shape[0], det_shape[1]).
    """
    # Combine single-track events to make multi-track events
    if tracks_per_event > 0:
        events = sum(generate_straight_tracks(n, det_shape)
                 for i in range(tracks_per_event))
        # Zero out the skipped layers
        events[:,0:skip_layers,:] = 0
    else:
        events = np.zeros((n,) + det_shape)
    return events

def sim_trap_straight_track(m, b, det_widths):
    """
    Simulate var-layer detector data for one straight track.
    Parameters:
        m: track slope parameter
        b: track first-layer intercept parameter
        det_shape: list of detector layer widths
    Returns:
        List of ndarrays representing the detector data from each layer.
    """
    data = [np.zeroes(width) for width in det_widths]
    hit_idxs = [int(round(m*l + b)) for l in range(len(det_widths))]
    for layer, pixel in enumerate(hit_idxs):
        data[layer][pixel] = 1
    return data

def generate_trap_straight_track(det_widths):
    """
    Sample track parameters and simulate data for one straight track in the
    variable-layer detector.
    Parameters:
        det_shape: list of detector layer widths
    Returns:
        List of ndarrays representing the detector data from each layer.
    """
    rands = np.random.random_sample(2)
    entry, exit = rands * (det_widths[0]-1, det_widths[-1]-1)
    slope = (exit - entry) / len(widths)
    return sim_trap_straight_track(slope, entry, det_widths)
