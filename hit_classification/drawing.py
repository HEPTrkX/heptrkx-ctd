"""
This module contains utility code for drawing data
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from toydata import track_hit_coords

def draw_layer(ax, data, title=None, **kwargs):
    """Draw one detector layer as an image"""
    ax.imshow(data.T, interpolation='none', aspect='auto',
              cmap='jet', origin='lower', **kwargs)
    if title is not None:
        ax.set_title(title)

def draw_layers(event, ncols=5, truthx=None, truthy=None, figsize=(12,5)):
    """Draw each detector layer as a grid of images"""
    num_det_layers = event.shape[0]
    nrows = math.ceil(float(num_det_layers)/ncols)
    fig = plt.figure(figsize=figsize)
    for ilay in range(num_det_layers):
        ax = plt.subplot(nrows, ncols, ilay+1)
        title = 'layer %i' % ilay
        draw_layer(ax, event[ilay], title=title)
        ax.autoscale(False)
        if truthx is not None and truthy is not None:
            ax.plot(truthx[ilay]-0.5, truthy[ilay]-0.5, 'w+')
    plt.tight_layout()
    return fig

def draw_projections(event, truthx=None, truthy=None, figsize=(12,5)):
    """Draw the 2D projections of an event, Z-X and Z-Y"""
    fig = plt.figure(figsize=figsize)
    plt.subplot(121)
    kwargs = dict(interpolation='none', aspect='auto', origin='lower', cmap='jet')
    plt.imshow(event.sum(axis=1).T, **kwargs)
    plt.xlabel('detector layer')
    plt.ylabel('pixel')
    plt.autoscale(False)
    if truthy is not None:
        plt.plot(np.arange(event.shape[0]-0.5), truthy-0.5, 'w-')
    plt.subplot(122)
    plt.imshow(event.sum(axis=2).T, **kwargs)
    plt.xlabel('detector layer')
    plt.ylabel('pixel')
    plt.tight_layout()
    plt.autoscale(False)
    if truthx is not None:
        plt.plot(np.arange(event.shape[0]-0.5), truthx-0.5, 'w-')
    return fig

def draw_3d_event(event, sig_track=None, sig_params=None, prediction=None,
                  pred_threshold=0.1, pred_alpha=0.2,
                  xlabel='detector layer', ylabel='pixel x', zlabel='pixel y',
                  color_map='rainbow'):
    """
    Draw 3D visualization of an event, a signal track, and a model prediction.
    """
    # Lookup the requested color map
    cmap = cm.get_cmap(color_map)

    # Setup the Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim(0, event.shape[0]-1)
    ax.set_ylim(0, event.shape[1])
    ax.set_zlim(0, event.shape[2])

    # Draw the event hits
    if sig_track is not None:
        event = event - sig_track
    evhits = np.nonzero(event)
    ax.scatter(evhits[0], evhits[1]+0.5, evhits[2]+0.5)

    # Draw the signal track hits
    if sig_track is not None:
        sighits = np.nonzero(sig_track)
        ax.scatter(sighits[0], sighits[1]+0.5, sighits[2]+0.5,
                   c='r', marker='D')

    # Draw the signal track true intercepts
    if sig_params is not None:
        layer_idx = np.arange(event.shape[0])
        sigx, sigy = track_hit_coords(sig_params, layer_idx,
                                      as_type=np.float32)
        ax.plot(layer_idx, sigx, sigy, 'r')

    # Draw the predictions on each detector plane
    if prediction is not None:
        # Surface grid coordinates, including endpoints. Note that we transpose the
        # coordinate arrays so that the first dimension gives the coordinates along
        # the row in X, which matches the way I represent my model predictions.
        grid_idx = np.arange(event.shape[1]+1)
        gridy, gridx = np.meshgrid(grid_idx, grid_idx)
        for i in np.arange(event.shape[0]):
            colors = cmap(prediction[i])
            # Set the global transparency of the prediction plane
            colors[:,:,3] = pred_alpha
            # Disable predictions below threshold
            colors[prediction[i] < pred_threshold,:] = 0.
            ax.plot_surface(i, gridx, gridy, rstride=1, cstride=1,
                            facecolors=colors, shade=False)
    plt.tight_layout()
    return fig, ax

def draw_2d_event(event, title=None, mask_ranges=None, mask_style='w:',
                  tight=True, **kwargs):
    """
    Draw and format one 2D detector event with matplotlib.
    Params:
        event: data for one event in image format
        title: plot title
        mask_range: tuple of arrays, (lower, upper) defining a detector
            mask envelope that will be drawn on the display
        kwargs: additional keywords passed to pyplot.plot
    """
    plt.imshow(event.T, interpolation='none', aspect='auto',
               origin='lower', **kwargs)
    if title is not None:
        plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Pixel')
    plt.autoscale(False)
    if tight:
        plt.tight_layout()
    if mask_ranges is not None:
        plt.plot(mask_ranges[:,0], mask_style)
        plt.plot(mask_ranges[:,1], mask_style)

def draw_2d_input_and_pred(event_input, event_pred, figsize=(9,4),
                           cmap='jet', mask_ranges=None, mask_style='k:',):
    fig = plt.figure(figsize=figsize)
    plt.subplot(121)
    draw_2d_event(event_input, title='Input', cmap=cmap,
                  mask_ranges=mask_ranges, mask_style=mask_style)
    plt.subplot(122)
    draw_2d_event(event_pred, title='Model prediction', cmap=cmap,
                  mask_ranges=mask_ranges, mask_style=mask_style)
    return fig

def draw_train_history(history, draw_val=True, figsize=(12,5)):
    """Make plots of training and validation losses and accuracies"""
    fig = plt.figure(figsize=figsize)
    # Plot loss
    plt.subplot(121)
    plt.plot(history.epoch, history.history['loss'], label='Training set')
    if draw_val:
        plt.plot(history.epoch, history.history['val_loss'], label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.epoch, history.history['acc'], label='Training set')
    if draw_val:
        plt.plot(history.epoch, history.history['val_acc'], label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.title('Training accuracy')
    plt.legend(loc=0)
    plt.tight_layout()
    return fig