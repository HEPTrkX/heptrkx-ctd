"""Drawing utilities"""

import math

import matplotlib.pyplot as plt

def draw_event(event, title=None, mask_ranges=None, tight=True, **kwargs):
    """
    Draw and format one 1D detector event with matplotlib.
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
        plt.plot(mask_ranges[:,0], 'w:')
        plt.plot(mask_ranges[:,1], 'w:')

def draw_input_and_pred(event_input, event_pred, figsize=(9,4), mask_ranges=None):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    draw_event(event_input, title='Input', mask_ranges=mask_ranges)
    plt.subplot(122)
    draw_event(event_pred, title='Model prediction', mask_ranges=mask_ranges)

def draw_filter_vis(pics, n_columns=4, title=None):
    """Draws the visualized filter images in a grid.
        pics: list of numpy arrays to visualize"""
    n_rows = int( math.ceil( len(pics) * 1.0 / n_columns ) )
    f, axarr = plt.subplots(n_rows, n_columns, figsize=(4*n_columns,4*n_rows))
    for i, pic in enumerate(pics):
        cur_row = i / n_columns
        cur_col = i % n_columns
        axarr[cur_row, cur_col].imshow(pic.T, interpolation='none',
                aspect='auto', origin='lower')
    plt.autoscale(False)
    if title is not None:
        f.suptitle(title)
    plt.show()
