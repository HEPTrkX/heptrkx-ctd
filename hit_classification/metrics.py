"""
This module contains code for calculating metrics like prediction accuracy.
"""

import numpy as np

def top_predictions(preds):
    """
    Choose the highest scored pixel on each layer.
    Arguments
        preds: array of shape (num_event, num_det_layer, num_pixels)
    Returns a boolean array of the chosen pixels.
    This could be generalized to top-k by using argsort instead.
    """
    top_pixels = preds.argmax(axis=2)
    top_preds = np.zeros_like(preds, dtype=np.bool)
    layer_idx = np.arange(preds.shape[1])
    for ievt in range(preds.shape[0]):
        top_preds[ievt, layer_idx, top_pixels[ievt]] = 1
    return top_preds

def calc_hit_accuracy(preds, targets, num_seed_layers=0):
    """
    Calculate the accuracy of hit predictions.
    Currently just uses the top pixel prediction.
    Arguments
        preds: array of shape (num_event, num_det_layer, num_pixel)
        targets: array of same shape
        num_seed_layers: detector layers to ignore
    Returns a float.
    """
    preds, targets = preds[:,num_seed_layers:], targets[:,num_seed_layers:]
    # Choose the top predictions in each detector layer
    top_preds = top_predictions(preds)
    # Probably unsafe to convert targets directly to bool
    top_targets = targets.astype(np.bool)
    num_correct = np.logical_and(top_preds, top_targets).sum()
    num_preds = preds.shape[0] * preds.shape[1]
    return float(num_correct) / num_preds