import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
def plot_colortable(colors, names, ncols=4, sort_colors=False):
    """ 
    Args:
        colors (list of str): hex code of rgba colours
        names (list of names): names that corresponds to the colours
    Returns:
        plot showing colours and its associated name
    """
    assert len(colors) == len(names),"len of colours and names dont match"

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    
    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        colors = [(n,c) for n,c in zip(names,colors)]#{n:c for n,c in zip(names,colors)}
        sorted_colors = sorted(colors, key=lambda x: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(x[1]))))
        names = [i[0] for i in sorted_colors]
        colors = [i[1] for i in sorted_colors]
    
    else:
        # sort by names
        colors = [(n,c) for n,c in zip(names,colors)]#{n:c for n,c in zip(names,colors)}
        sorted_colors = sorted(colors, key=lambda x: x[0])
        names = [i[0] for i in sorted_colors]
        colors = [i[1] for i in sorted_colors]
        
    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[i], edgecolor='0.7')
        )
    
    plt.show()
    return 

import matplotlib as mpl
def get_colorbar(vmin,vmax,label="Units",cmap="plasma",orientation="horizontal", plot=True):
    """ 
    Args:
        vmin (float): smallest value to be mapped to the start of color map
        vmax (float): largest value to be mapped to the end of color map
        label (str): label for colorbar e.g. units
        cmap (str): Name of the matplotlib colormap from which to choose the colors.
        orientation (str): orientation of colorbar e.g. horizontal, vertical
        plot (bool): if plot is True, plot colorbar
        ax (Ax or None): if ax is not None, plot colorbar next to existing Ax
    """

    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin, vmax)
    cbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # cax = ax (meaning "draw the colorbar on ax")
    # ax = ax (display the colorbar next to a pre-existing Axes ax)
    if plot:
        fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')
        fig.colorbar(cbar,cax=ax, orientation=orientation, label=label)
    # else:
    #     fig.colorbar(cbar,ax=ax, orientation=orientation, label=label)
    return cbar

# reorder legend by row instead of column
reorder_legend = lambda l, nc: sum((l[i::nc] for i in range(nc)), []) #hl = handles/lanels, nc=number of columns
# usage example
# fig.legend(handles=reorder(handles,n_assum),loc=loc, ncol=n_assum, fontsize='medium')