"""
This module contains helper code for generating and manipulating toy detector
data for the ML algorithms.

The toy model has 2D square detector layers.
"""

from __future__ import print_function

import numpy as np

def gen_noise(shape, prob=0.1, seed_layers=0):
    """Generate uniform noise data of requested shape"""
    noise = (np.random.random_sample(shape) < prob).astype(np.int8)
    noise[:,:seed_layers,:,:] = 0
    return noise

def sample_track_params(n, num_det_layers, det_layer_size):
    """Generate track parameters constrained within detector shape"""
    # Sample the entry and exit points for tracks
    entry_points = np.random.uniform(0, det_layer_size, size=(n, 2))
    exit_points = np.random.uniform(0, det_layer_size, size=(n, 2))    
    # Calculate slope parameters
    slopes = (exit_points - entry_points) / float(num_det_layers - 1)
    return np.concatenate([slopes, entry_points], axis=1)

def track_hit_coords(params, det_layer_idx=None, num_det_layers=None, as_type=np.int):
    """
    Given an array of track params, give the coordinates
    of the hits in detector index space
    """
    if det_layer_idx is None:
        det_layer_idx = np.arange(num_det_layers)
    xslope, yslope, xentry, yentry = params
    xhits = xslope*det_layer_idx + xentry
    yhits = yslope*det_layer_idx + yentry
    return xhits.astype(as_type), yhits.astype(as_type)

def gen_straight_tracks(n, num_det_layers, det_layer_size):
    """Generate n straight tracks"""
    # Initialize the data
    data = np.zeros((n, num_det_layers, det_layer_size, det_layer_size),
                    dtype=np.float32)
    # Sample track parameters
    params = sample_track_params(n, num_det_layers, det_layer_size)
    # Calculate hit positions and fill hit data
    idx = np.arange(num_det_layers)
    for ievt in range(n):
        xhits, yhits = track_hit_coords(params[ievt], idx)
        data[ievt,idx,xhits,yhits] = 1   
    return data, params

def gen_bkg_tracks(num_event, num_det_layers, det_layer_size,
                   avg_bkg_tracks=3, seed_layers=0):
    """
    Generate background tracks in the non-seed detector layers.
    Samples the number of tracks for each event from a poisson
    distribution with specified mean avg_bkg_tracks.
    """
    num_bkg_tracks = np.random.poisson(avg_bkg_tracks, num_event)
    bkg_tracks = np.zeros((num_event, num_det_layers, det_layer_size, det_layer_size),
                          dtype=np.float32)
    for ievt in range(num_event):
        ntrk = num_bkg_tracks[ievt]
        bkg_tracks[ievt] = sum(gen_straight_tracks(ntrk, num_det_layers, det_layer_size)[0])
    bkg_tracks[:,:seed_layers,:,:] = 0
    return bkg_tracks

def generate_data(shape, num_seed_layers=3, avg_bkg_tracks=3,
                  noise_prob=0.01, verbose=True):
    """
    Top level function to generate a dataset.
    
    Returns arrays (events, sig_tracks, sig_params)
    """
    num_event, num_det_layers, det_layer_size, _ = shape
    # Signal tracks
    sig_tracks, sig_params = gen_straight_tracks(
        num_event, num_det_layers, det_layer_size)
    # Background tracks
    bkg_tracks = gen_bkg_tracks(
        num_event, num_det_layers, det_layer_size,
        avg_bkg_tracks=avg_bkg_tracks, seed_layers=num_seed_layers)
    # Noise
    noise = gen_noise(shape, prob=noise_prob, seed_layers=num_seed_layers)
    # Full events
    events = sig_tracks + bkg_tracks + noise
    events[events > 1] = 1
    # Print data sizes
    if verbose:
        print('Sizes of arrays')
        print('  events:     %g MB' % (events.dtype.itemsize * events.size / 1e6))
        print('  sig_tracks: %g MB' % (sig_tracks.dtype.itemsize * sig_tracks.size / 1e6))
        print('  bkg_tracks: %g MB' % (bkg_tracks.dtype.itemsize * bkg_tracks.size / 1e6))
        print('  noise:      %g MB' % (noise.dtype.itemsize * noise.size / 1e6))
        print('  sig_params: %g MB' % (sig_params.dtype.itemsize * sig_params.size / 1e6))
    return events, sig_tracks, sig_params