"""
plot_util.py

This module contains basic functions for plotting with pyplot.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import colorsys
from pathlib import Path
import logging
import warnings

import matplotlib as mpl
import matplotlib.cm as mpl_cm
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from util import file_util, gen_util, logger_util, math_util, rand_util

logger = logging.getLogger(__name__)

TAB = "    "

LINCLAB_COLS = {"blue"  : "#50a2d5", # Linclab blue
                "red"   : "#eb3920", # Linclab red
                "gray"  : "#969696", # Linclab gray
                "green" : "#76bb4b", # Linclab green
                "purple": "#9370db",
                "orange": "#ff8c00",
                "pink"  : "#bb4b76",
                "yellow": "#e0b424",
                "brown" : "#b04900",
                }


#############################################
def linclab_plt_defaults(font="Liberation Sans", fontdir=None, 
                         log_fonts=False, example=False, dirname=".", 
                         **cyc_args):
    """
    linclab_plt_defaults()

    Sets pyplot defaults to Linclab style.

    Optional args:
        - font (str or list): font (or font family) to use, or list in order of 
                              preference
                              default: "Liberation Sans"
        - fontdir (Path)    : directory to where extra fonts (.ttf) are stored
                              default: None
        - log_fonts (bool)  : if True, an alphabetical list of available fonts 
                              is logged
                              default: False
        - example (bool)    : if True, an example plot is created and saved
                              default: False
        - dirname (Path)    : directory in which to save example if example is 
                              True 
                              default: "."

    Kewyord args:
        - cyc_args (dict): keyword arguments for plt.cycler()
    """

    col_order = ["blue", "red", "gray", "green", "purple", "orange", "pink", 
                 "yellow", "brown"]
    colors = [get_color(key) for key in col_order] 
    col_cyc = plt.cycler(color=colors, **cyc_args)

    # set pyplot params
    params = {"axes.labelsize"       : "xx-large", # xx-large axis labels
              "axes.linewidth"       : 1.5,        # thicker axis lines
              "axes.prop_cycle"      : col_cyc,    # line color cycle
              "axes.spines.right"    : False,      # no axis spine on right
              "axes.spines.top"      : False,      # no axis spine at top
              "axes.titlesize"       : "xx-large", # xx-large axis title
              "errorbar.capsize"     : 4,          # errorbar cap length
              "figure.titlesize"     : "xx-large", # xx-large figure title
              "figure.autolayout"    : True,       # adjusts layout
              "font.size"            : 12,         # basic font size value
              "legend.fontsize"      : "x-large",  # x-large legend text
              "lines.dashed_pattern" : [8.0, 4.0], # longer dashes
              "lines.linewidth"      : 2.5,        # thicker lines
              "lines.markeredgewidth": 2.5,        # thick marker edge widths 
                                                   # (e.g., cap thickness) 
              "lines.markersize"     : 10,         # bigger markers
              "patch.linewidth"      : 2.5,        # thicker lines for patches
              "savefig.format"       : "svg",      # figure save format
              "savefig.bbox"         : "tight",    # tight cropping of figure
              "xtick.labelsize"      : "x-large",  # x-large x-tick labels
              "xtick.major.size"     : 8.0,        # longer x-ticks
              "xtick.major.width"    : 2.0,        # thicker x-ticks
              "ytick.labelsize"      : "x-large",  # x-large y-tick labels
              "ytick.major.size"     : 8.0,        # longer y-ticks
              "ytick.major.width"    : 2.0,        # thicker y-ticks
              }


    set_font(font, fontdir, log_fonts)

    # update pyplot parameters
    plt.rcParams.update(params)

    # create and save an example plot, if requested
    if example:
        fig, ax = plt.subplots(figsize=[8, 8])
        
        n_col = len(colors)
        x = np.asarray(list(range(10)))[:, np.newaxis]
        y = np.repeat(x/2., n_col, axis=1) - \
            np.asarray(list(range(-n_col, 0)))[np.newaxis, :]
        ax.plot(x, y)

        # label plot
        legend_labels = [f"{name}: {code}" 
            for name, code in zip(col_order, colors)]
        ax.legend(legend_labels)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Example plot", y=1.02)
        ax.axvline(x=1, ls="dashed", c="k")
    
        dirname = Path(dirname)
        dirname.mkdir(parents=True, exist_ok=True)
        
        ext = plt.rcParams["savefig.format"]
        savepath = dirname.joinpath(f"example_plot").with_suffix(f".{ext}")
        fig.savefig(savepath)

        logger.info(f"Example saved under {savepath}")


#############################################
def linclab_colormap(nbins=100, gamma=1.0, no_white=False):
    """
    linclab_colormap()

    Returns a matplotlib colorplot using the linclab blue, white and linclab 
    red.
    
    Optional args:
        - nbins (int)    : number of bins to use to create colormap
                           default: 100
        - gamma (num)    : non-linearity
                           default: 1.0
        - no_white (bool): if True, white as the intermediate color is omitted 
                           from the colormap.
                           default: False
    Returns:
        - cmap (colormap): a matplotlib colormap
    """

    colors = [LINCLAB_COLS["blue"], "#ffffff", LINCLAB_COLS["red"]]
    name = "linclab_bwr"
    if no_white:
        colors = [colors[0], colors[-1]]
        name = "linclab_br"

    # convert to RGB
    rgb_col = [[] for _ in range(len(colors))]
    for c, col in enumerate(colors):
        ch_vals = mpl.colors.to_rgb(col)
        for ch_val in ch_vals:
            rgb_col[c].append(ch_val)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        name, rgb_col, N=nbins, gamma=gamma)

    cmap.set_bad(color="black")

    return cmap


#############################################
def nipy_spectral_no_white_cmap(nbins=100, gamma=1.0):
    """
    nipy_spectral_no_white_cmap()

    Returns the nipy_spectral matplotlib colormap adjusted to not have the 
    white upper end.

    Optional args:
        - nbins (int): number of bins to use to create colormap
                       default: 100
        - gamma (num): non-linearity
                       default: 1.0

    Returns:
        - cmap (colormap): a matplotlib colormap
    """

    colors = mpl_cm.nipy_spectral(np.linspace(0, 1, 11))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nipy_spectral_no_white", colors[:-1], N=nbins, gamma=gamma)

    return cmap


#############################################
def viridis_with_black_cmap(nbins=100, gamma=1.0):
    """
    viridis_with_black_cmap()

    Returns the viridis matplotlib colormap adjusted to have a black lower end.

    Optional args:
        - nbins (int): number of bins to use to create colormap
                       default: 100
        - gamma (num): non-linearity
                       default: 1.0

    Returns:
        - cmap (colormap): a matplotlib colormap
    """

    colors = mpl_cm.viridis(np.linspace(0, 1, 9))
    colors = np.insert(colors, 0, np.asarray([0, 0, 0, 1]), axis=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "viridis_with_black", colors, N=nbins, gamma=gamma)

    return cmap


#############################################
def set_font(font="Liberation Sans", fontdir=None, log_fonts=False):
    """
    set_font()

    Sets pyplot font to preferred values.
    
    NOTE: This function is particular convoluted to enable to clearest warnings 
    when preferred fonts/font families are not found.

    Optional args:
        - font (str or list): font or font family to use, or list in order of 
                              preference
                              default: "Liberation Sans"
        - fontdir (Path)    : directory to where extra fonts (.ttf) are stored
                              default: None
        - log_fonts (bool)  : if True, an alphabetical list of available fonts 
                              is logged
                              default: False
    """

    # keep in lower case
    font_families = ["cursive", "family", "fantasy", "monospace", 
        "sans-serif", "serif"]

    if fontdir is not None:
        fontdir = Path(fontdir)
        if not fontdir.exists():
            raise OSError(f"{fontdir} font directory does not exist.")

        # add new fonts to list of available fonts if a font directory is provided
        fontdirs = [fontdir, ]
        # prevent a long stream of debug messages
        logging.getLogger("matplotlib.font_manager").disabled = True
        font_files = fm.findSystemFonts(fontpaths=fontdirs)
        for font_file in font_files:
            fm.fontManager.addfont(font_file)
    
    # compile list of available fonts/font families 
    # (includes checking each font family to see if any of its fonts are available)
    all_fonts = list(set([f.name for f in fm.fontManager.ttflist]))
    all_fonts_lower = [font.lower() for font in all_fonts]
    font_families_found = []
    none_found_str = " (no family fonts found)"
    for f, font_family in enumerate(font_families):
        available_fonts = list(filter(lambda x: x.lower() in all_fonts_lower, 
            plt.rcParams[f"font.{font_family}"]))
        if len(available_fonts) == 0:
            font_families_found.append(
                f"{font_family}{none_found_str}")
        else:
            font_families_found.append(font_family)

    # log list of font families and fonts, if requested
    if log_fonts:
        font_log = ""
        for i, (font_type, font_list) in enumerate(zip(
            ["Font families", "Available fonts"], [font_families, all_fonts])):
            sep = "" if i == 0 else "\n\n"
            sorted_fonts_str = f"\n{TAB}".join(sorted(font_list))
            font_log = (f"{font_log}{sep}{font_type}:"
                f"\n{TAB}{sorted_fonts_str}")
        logger.info(font_log)
    
    # compile ordered list of available fonts/font families, in the preferred 
    # order to use in setting the mpl font choice parameter.
    fonts = gen_util.list_if_not(font)
    params = {
        "font.family": plt.rcParams["font.family"]
    }
    fonts_idx_added = []
    for f, font in enumerate(fonts):
        if font.lower() in all_fonts_lower:
            font_name = all_fonts[all_fonts_lower.index(font.lower())]
        elif (font.lower() in font_families_found and 
            none_found_str not in font.lower()):
            font_name = font.lower()
        else:
            if font.lower() in font_families:
                fonts[f] = f"{font} family fonts"
            continue

        # if found, add/move to correct position in list
        if font_name in params["font.family"]:
            params["font.family"].remove(font_name)

        params["font.family"].insert(len(fonts_idx_added), font_name)
        fonts_idx_added.append(f)

    #  warn if the first (set of) requested fonts/font families were not found.
    if len(fonts_idx_added) == 0:
        first_font_added = None
    else: 
        first_font_added = min(fonts_idx_added)
    if first_font_added != 0:
        omitted_str = ", ".join(fonts[: first_font_added])
        selected_str = ""
        if len(plt.rcParams["font.family"]) != 0:
            selected = plt.rcParams["font.family"][0]
            selected_str = f"\nFont set to {selected}."
            if selected in font_families:
                selected_str = selected_str.replace(".", " family.")
        warnings.warn(f"Requested font(s) not found: {omitted_str}."
            f"{selected_str}", category=UserWarning, stacklevel=1)
    
    plt.rcParams.update(params)

    return


#############################################
def get_color(col="red", ret="single"):
    """
    get_color()

    Returns requested info for the specified color.

    Optional args:
        - col (str): color for which to return info
                     default: "red"
        - ret (str): type of information to return for color
                     default: "single"

    Returns:
        if ret == "single" or "both":
        - single (str)   : single hex code corresponding to requested color
        if ret == "col_ends" or "both":
        - col_ends (list): hex codes for each end of a gradient corresponding to 
                           requested color
    """
    
    # list of defined colors
    curr_cols = ["blue", "red", "gray", "green", "purple", "orange", "pink", 
        "yellow", "brown"]
    
    if col == "blue":
        # cols  = ["#7cc7f9", "#50a2d5", "#2e78a9", "#16547d"]
        col_ends = ["#8DCCF6", "#07395B"]
        single   = LINCLAB_COLS["blue"]
    elif col == "red":
        # cols = ["#f36d58", "#eb3920", "#c12a12", "#971a07"]
        col_ends = ["#EF6F5C", "#7D1606"]
        single   = LINCLAB_COLS["red"]
    elif col in ["gray", "grey"]:
        col_ends = ["#969696", "#060707"]
        single   = LINCLAB_COLS["gray"]
    elif col == "green":
        col_ends = ["#B3F38E", "#2D7006"]
        single   = LINCLAB_COLS["green"]
    elif col == "purple":
        col_ends = ["#B391F6", "#372165"]
        single   = LINCLAB_COLS["purple"]
    elif col == "orange":
        col_ends = ["#F6B156", "#CD7707"]
        single   = LINCLAB_COLS["orange"]
    elif col == "pink":
        col_ends = ["#F285AD", "#790B33"]
        single   = LINCLAB_COLS["pink"]
    elif col == "yellow":
        col_ends = ["#F6D25D", "#B38B08"]
        single   = LINCLAB_COLS["yellow"]
    elif col == "brown":
        col_ends = ["#F7AD75", "#7F3904"]
        single   = LINCLAB_COLS["brown"]
    else:
        gen_util.accepted_values_error("col", col, curr_cols)

    if ret == "single":
        return single
    elif ret == "col_ends":
        return col_ends
    elif ret == "both":
        return single, col_ends
    else:
        gen_util.accepted_values_error(
            "ret", ret, ["single", "col_ends", "both"])


#############################################
def get_color_range(n=4, col="red"):
    """
    get_color_range()

    Returns a list of color values around the specified general color requested.

    Optional args:
        - n (int)          : number of colors required
                             default: 4
        - col (str or list): general color or two colors (see get_color() for 
                             accepted colors)
                             default: "red"

    Returns:
        - cols (list): list of colors
    """

    
    cols = gen_util.list_if_not(col)
    if len(cols) not in [1, 2]:
        raise ValueError("'col' must be of length one or two")
    if len(cols) == 2 and n == 1:
        cols = cols[0:1] # retain only first colour

    ends = []
    for col in cols:
        single, col_ends = get_color(col, ret="both")
        if len(cols) == 2:
            ends.append(single)
        else:
            ends = col_ends

    if n == 1:
        cols = [single]
    else:
        cols = get_col_series(ends, n)

    return cols


#############################################
def get_hex_color_range(n=3, col="#0000FF", interval=0.3):
    """
    get_hex_color_range()

    Returns a list of color values from dark to light, around the specified 
    hex color requested.

    Optional args:
        - n (int)          : number of colors required
                             default: 3
        - col (str or list): color in hex (or matplotlib name)
                             default: "#0000FF"
        - interval (float) : space between each color, centered around col
                             default: 0.3

    Returns:
        - cols (list): list of colors
    """

    # identify range of values to use for scaling lightness
    full_interval = interval * (n - 1)
    range_scale = np.linspace(1 - full_interval / 2, 1 + full_interval / 2, n)

    r, g, b = mpl.colors.to_rgb(col)

    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    cols = []
    for scale_l in range_scale:
        new_col = colorsys.hls_to_rgb(h, max(0, min(1, l * scale_l)), s = s)
        cols.append(new_col)

    return cols


#############################################
def manage_mpl(plt_bkend=None, linclab=True, fontdir=None, cmap=False, 
               nbins=100):
    """
    manage_mpl()

    Makes changes to the matplotlib backend used as well as matplotlib plotting
    defaults. If cmap is True, a colormap is returned.

    Optional args:
        - plt_bkend (str): matplotlib backend to use
                           default: None
        - linclab (bool) : if True, the Linclab default are set
                           default: True
        - fontdir (Path) : directory to where extra fonts (.ttf) are stored
                           default: None
        - cmap (bool)    : if True, a colormap is returned. If linclab is True,
                           the Linclab colormap is returned, otherwise the 
                           "viridis" colormap
                           default: False
        - nbins (int)    : number of bins to use to create colormap
                           default: 100

    Returns:
        if cmap:
        - cmap (colormap): a matplotlib colormap
    """

    if plt_bkend is not None:
        plt.switch_backend(plt_bkend)
    
    if linclab:
        linclab_plt_defaults(
            font=["Arial", "Liberation Sans"], fontdir=fontdir)

    if cmap:
        if linclab:
            cmap = linclab_colormap(nbins)
        else:
            cmap = "viridis"
        return cmap


#############################################
def remove_axis_marks(sub_ax):
    """
    remove_axis_marks(sub_ax)

    Removes all axis marks (ticks, tick labels, spines).

    Required args:
        - sub_ax (plt Axis subplot): subplot    
    """

    sub_ax.tick_params(axis="x", which="both", bottom=False, top=False) 
    sub_ax.tick_params(axis="y", which="both", left=False, right=False) 

    sub_ax.set_xticks([])
    sub_ax.set_yticks([])

    for spine in ["right", "left", "top", "bottom"]:
        sub_ax.spines[spine].set_visible(False)


#############################################
def set_ticks_safe(sub_ax, ticks, axis="x"):
    """
    set_ticks_safe(sub_ax, ticks)

    Sets specified ticks on specified axis, allowing ticks to be a single value. 

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - ticks (num or array-like): tick values

    Optional args:
        - axis (str): axis for which to set ticks, i.e., x, y or both
                      default: "x"
    """

    if isinstance(ticks, np.ndarray):
        if len(ticks.shape) == 0:
            ticks = ticks.reshape(1)
    else:
        ticks = gen_util.list_if_not(ticks)

    if axis == "x":
        sub_ax.set_xticks(ticks)
    elif axis == "y":
        sub_ax.set_yticks(ticks)
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])
        
    
#############################################
def set_ticks(sub_ax, axis="x", min_tick=0, max_tick=1.5, n=6, pad_p=0.05, 
              minor=False):
    """
    set_ticks(sub_ax)

    Sets ticks on specified axis and axis limits around ticks using specified 
    padding. 

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - axis (str)    : axis for which to set ticks, i.e., x, y or both
                          default: "x"
        - min_tick (num): first tick value
                          default: 0
        - max_tick (num): last tick value
                          default: 1.5
        - n (int)       : number of ticks
                          default: 6
        - pad_p (num)   : percentage to pad axis length
                          default: 0.05
        - minor (bool)  : if True, minor ticks are included
                          default: False
    """

    pad = (max_tick - min_tick) * pad_p
    min_end = min_tick - pad
    max_end = max_tick + pad

    if axis == "both":
        axis = ["x", "y"]
    elif axis in ["x", "y"]:
        axis = gen_util.list_if_not(axis)
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y", "both"])

    if "x" in axis:
        if min_end != max_end:
            sub_ax.set_xlim(min_end, max_end)
            sub_ax.set_xticks(np.linspace(min_tick, max_tick, n), minor=minor)
        else:
            sub_ax.set_xticks([min_end])
    elif "y" in axis:
        if min_end != max_end:
            sub_ax.set_ylim(min_end, max_end)
            sub_ax.set_yticks(np.linspace(min_tick, max_tick, n), minor=minor)
        else:
            sub_ax.set_yticks([min_end])


#############################################
def set_ticks_from_vals(sub_ax, vals, axis="x", n=6, pad_p=0.05):
    """
    set_ticks_from_vals(sub_ax, vals)

    Sets ticks on specified axis and axis limits around ticks using specified 
    padding, based on the plotted axis values. 

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - vals (array-like)        : axis values in the data

    Optional args:
        - axis (str )     : axis for which to set ticks, i.e., x, y or both
                            default: "x"
        - n (int)         : number of ticks
                            default: 6
        - pad_p (num)     : percentage to pad axis length
                            default: 0.05
        - ret_ticks (bool): if True, tick values are
    """

    n_ticks = np.min([n, len(vals)])
    diff = np.max(vals) - np.min(vals)
    if diff == 0:
        n_dig = 0
    else:
        n_dig = - np.floor(np.log10(np.absolute(diff))).astype(int) + 1
    set_ticks(sub_ax, axis, np.around(np.min(vals), n_dig), 
        np.around(np.max(vals), n_dig), n_ticks)


#############################################
def get_subax(ax, i):
    """
    get_subax(ax, i)

    Returns the correct sub_ax from a 1D or 2D axis array based on a 1D index. 
    Indexing is by column, then row.

    Required args:
        - ax (plt Axis): axis
        - i (int)      : 1D subaxis index

    Returns:
        - sub_ax (plt Axis subplot): subplot
    """

    if len(ax.shape) == 1:
        n = ax.shape[0]
        sub_ax = ax[i % n]
    else:
        ncols = ax.shape[1]
        sub_ax = ax[i // ncols][i % ncols]

    return sub_ax

#############################################
def turn_off_extra(ax, n_plots):
    """
    turn_off_extra(ax, i)

    Turns off axes of subplots beyond the number of plots. Assumes that the 
    subplots are filled first by column and second by row.

    Required args:
        - ax (plt Axis): axis
        - n_plots (int): number of plots used (consecutive)
    """

    if not (n_plots < ax.size):
        return
    
    for i in range(n_plots, ax.size):
        sub_ax = get_subax(ax, i)
        sub_ax.set_axis_off()


#############################################
def share_lims(ax, axis="row"):
    """
    share_lims(ax)

    Adjusts limits within rows or columns for a 2D axis array. 

    Required args:
        - ax (plt Axis): axis (2D array)

    Optional args:
        - axis (str): which axis to match limits along
                      default: "row"
    """

    if len(ax.shape) != 2:
        raise NotImplementedError("Function only implemented for 2D axis "
                                  "arrays.")
    
    if axis == "row":
        for r in range(ax.shape[0]):
            ylims = [np.inf, -np.inf]
            for task in ["get", "set"]:    
                for c in range(ax.shape[1]):
                    if task == "get":
                        lim = ax[r, c].get_ylim()
                        if lim[0] < ylims[0]:
                            ylims[0] = lim[0]
                        if lim[1] > ylims[1]:
                            ylims[1] = lim[1]
                    elif task == "set":
                        ax[r, c].set_ylim(ylims)
    
    if axis == "col":
        for r in range(ax.shape[1]):
            xlims = [np.inf, -np.inf]
            for task in ["get", "set"]:   
                for c in range(ax.shape[0]):
                    if task == "get":
                        lim = ax[r, c].get_xlim()
                        if lim[0] < xlims[0]:
                            xlims[0] = lim[0]
                        if lim[1] > xlims[1]:
                            xlims[1] = lim[1]
                    elif task == "set":
                        ax[r, c].set_xlim(xlims)


#############################################
def set_axis_digits(sub_ax, xaxis=None, yaxis=None):
    """
    set_axis_digits(sub_ax)

    Sets the number of digits in the axis tick labels.

    Required args:
        - sub_ax (plt Axis subplot): subplot
    
    Optional args:
        - xaxis (int): number of digits for the x axis 
                       default: None
        - yaxis (int): number of digits for the y axis
                       default: None
    """

    if xaxis is not None:
        n_dig_str = f"%.{int(xaxis)}f"
        sub_ax.xaxis.set_major_formatter(FormatStrFormatter(n_dig_str))

    if yaxis is not None:
        n_dig_str = f"%.{int(yaxis)}f"
        sub_ax.yaxis.set_major_formatter(FormatStrFormatter(n_dig_str))


#############################################
def remove_ticks(sub_ax, xaxis=True, yaxis=True):
    """
    remove_ticks(sub_ax)

    Removes ticks and tick labels for the specified axes.

    Required args:
        - sub_ax (plt Axis subplot): subplot
    
    Optional args:
        - xaxis (bool): if True, applies to x axis 
                        default: None
        - yaxis (bool): if True, applies to y axis
                       default: None
    """

    if xaxis:
        sub_ax.tick_params(axis="x", which="both", bottom=False) 
        sub_ax.set_xticks([])
    if yaxis:
        sub_ax.tick_params(axis="y", which="both", bottom=False) 
        sub_ax.set_yticks([])


#############################################
def remove_graph_bars(sub_ax, bars="all"):
    """
    remove_graph_bars(sub_ax)

    Removes the framing bars around a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
    
    Optional args:
        - bars (str or list): bars to remove ("all", "vert", "horiz" or a list 
                              of bars (amongst "top", "bottom", "left", 
                              "right"))
                              default: "all"
    """

    if isinstance(bars, list):
        for bar in bars:
            sub_ax.spines[bar].set_visible(False)

    else: 
        if bars in ["top", "bottom", "right", "left"]:
            keys = [bars]
        if bars == "all":
            keys = sub_ax.spines.keys()
        elif bars == "vert":
            keys = ["left", "right"]
        elif bars == "horiz":
            keys = ["top", "bottom"]    
        for key in keys:
            sub_ax.spines[key].set_visible(False)


#############################################
def init_fig(n_subplots, ncols=3, sharex=False, sharey=True, subplot_hei=7, 
             subplot_wid=7, gs=None, proj=None, **fig_kw):
    """
    init_fig(n_subplots, fig_par)

    Returns a figure and axes with the correct number of rows and columns for 
    the number of subplots, following the figure parameters.

    Required args:
        - n_subplots (int) : number of subplots to accomodate in the figure
        
    Optional args:
        - ncols (int)      : number of columns in the figure
                             default: 3
        - sharex (bool)    : if True, x axis lims are shared across subplots
                             default: False
        - sharey (bool)    : if True, y axis lims are shared across subplots
                             default: True
        - subplot_hei (num): height of each subplot (inches)
                             default: 7
        - subplot_wid (num): width of each subplot (inches)
                             default: 7
        - gs (dict)        : plt gridspec dictionary
                             default: None
        - proj (str)       : plt projection argument (e.g. "3d")
                             default: None
 
    Kewyord args:
        - fig_kw (dict): keyword arguments for plt.subplots()

    Returns:
        - fig (plt Fig): fig
        - ax (plt Axis): axis (even if for just one subplot)
    """
   
    nrows = 1
    if n_subplots == 1:
        ncols = 1
    elif n_subplots < ncols:
        ncols = n_subplots
    else:
        nrows = int(np.ceil(n_subplots/float(ncols)))
        # find minimum number of columns given number of rows
        ncols = int(np.ceil(n_subplots/float(nrows)))

    fig, ax = plt.subplots(
        ncols=ncols, nrows=nrows, 
        figsize=(ncols*subplot_wid, nrows*subplot_hei), sharex=sharex, 
        sharey=sharey, squeeze=False, gridspec_kw=gs, 
        subplot_kw={"projection": proj}, **fig_kw)

    return fig, ax


#############################################
def savefig(fig, savename, fulldir=".", datetime=True, use_dt=None, 
            fig_ext="svg", overwrite=False, save_fig=True, log_dir=True, 
            **savefig_kw):
    """
    savefig(fig, savename)

    Saves a figure under a specific directory and name, following figure
    parameters and returns final directory name.

    Required args:
        - fig (plt Fig) : figure (if None, no figure is saved, but fulldir is 
                          created and name is returned)
        - savename (str): name under which to save figure
    
    Optional args:
        - fulldir (Path)  : directory in which to save figure
                            default: "."
        - datetime (bool) : if True, figures are saved in a subfolder named 
                            based on the date and time.
                            default: True
        - use_dt (str)    : datetime folder to use
                            default: None
        - fig_ext (str)   : figure extension
                            default: "svg"
        - overwrite (bool): if False, overwriting existing figures is prevented 
                            by adding suffix numbers.
                            default: False        
        - save_fig (bool) : if False, the figure saving step is skipped. If 
                            log_dir, figure directory will still be logged. 
                            default: True
        - log_dir (bool)  : if True, the save directory is logged 
                            default: True

    Kewyord args:
        - savefig_kw (dict): keyword arguments for plt.savefig()

    Returns:
        - fulldir (Path): final name of the directory in which the figure is 
                          saved (may differ from input fulldir, if datetime 
                          subfolder is added.)
    """

    # add subfolder with date and time
    fulldir = Path(fulldir)
    if datetime:
        if use_dt is not None:
            fulldir = fulldir.joinpath(use_dt)
        else:
            datetime = gen_util.create_time_str()
            fulldir = fulldir.joinpath(datetime)

    # create directory if doesn't exist
    file_util.createdir(fulldir, log_dir=False)

    if fig is not None:
        # get extension and savename
        if overwrite:
            fullname, _ = file_util.add_ext(savename, fig_ext) 
        else:
            fullname = file_util.get_unique_path(
                savename, fulldir, ext=fig_ext
                ).parts[-1]
        if save_fig:
            fig.savefig(fulldir.joinpath(fullname), **savefig_kw)
            log_text = "Figures saved under"
        else:
            log_text = "Figure directory (figure not saved):"

        if log_dir:
            logger.info(f"{log_text} {fulldir}.", extra={"spacing": "\n"})
            
    return fulldir


#############################################
def get_repeated_bars(xmin, xmax, cycle=1.0, offset=0):
    """
    get_repeated_bars(xmin, xmax)

    Returns lists of positions at which to place bars cyclicly.

    Required args:
        - xmin (num): minimum x value
        - xmax (num): maximum x value (included)

    Optional args:
        - cycle (num) : distance between bars
                        default: 1.0
        - offset (num): position of reference bar 
                        default: 0
    
    Returns:
        - bars (list) : list of x coordinates at which to add bars
                        
    """

    min_bar = np.ceil((xmin - offset) / cycle).astype(int)
    max_bar = np.floor((xmax - offset) / cycle).astype(int) + 1 # excluded
    bars = [cycle * b + offset for b in range(min_bar, max_bar)]

    return bars


#############################################
def add_labels(sub_ax, labels, xpos, t_hei=0.9, color="k", fontsize=18, 
               ha="center", **text_kw):
    """
    add_labels(sub_ax, labels, xpos)

    Adds labels to a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - labels (list or str)     : list of labels to add to axis
        - xpos (list or num)       : list of x coordinates at which to add 
                                     labels (same length as labels)
      
    Optional args:
        - t_hei (num)   : height relative to y limits at which to place labels. 
                          default: 0.9
        - color (str)   : text color
                          default: "k"
        - fontsize (int): fontsize
                          default: 18
        - ha (str)      : text horizontal alignment
                          default: "center"

    Kewyord args:
        - text_kw (dict): keyword arguments for plt.text()
    """

    labels = gen_util.list_if_not(labels)
    xpos = gen_util.list_if_not(xpos)

    if len(labels) != len(xpos):
        raise ValueError("Arguments 'labels' and 'xpos' must be of "
            "the same length.")

    ymin, ymax = sub_ax.get_ylim()
    ypos = (ymax - ymin) * t_hei + ymin
    for l, x in zip(labels, xpos):
        sub_ax.text(x, ypos, l, ha=ha, fontsize=fontsize, color=color)


#############################################
def add_bars(sub_ax, hbars=None, bars=None, color="k", alpha=0.5, ls="dashed", 
             lw=None, **axline_kw):
    """
    add_bars(sub_ax)

    Adds dashed vertical bars to a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - hbars (list or num): list of x coordinates at which to add 
                               heavy dashed vertical bars
                               default: None
        - bars (list or num) : list of x coordinates at which to add 
                               dashed vertical bars
                               default: None
        - color (str)        : color to use
                               default: "k"
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - ls (str or tuple)  : linestyle
                               default: "dashed"
        - lw (list)          : list of 2 linewidths [hbars, bars]
                               (if None, default values are used)
                               default: None

    Kewyord args:
        - axline_kw (dict): keyword arguments for plt.axvline() or plt.axhline()
    """

    if lw is None:
        lw = [2.5, 1.5]
    else:
        lw = gen_util.list_if_not(lw)
        if len(lw) == 1:
            lw = lw * 2

    if len(lw) != 2:
        raise ValueError("'lw' must be a list of length 2.")

    torem = []
    if hbars is not None:
        hbars = gen_util.list_if_not(hbars)
        torem = hbars
        for b in hbars:
            sub_ax.axvline(x=b, ls=ls, color=color, lw=lw[0], alpha=alpha, 
                           **axline_kw)
    if bars is not None:
        bars = gen_util.remove_if(bars, torem)
        for b in bars:
            sub_ax.axvline(x=b, ls=ls, color=color, lw=lw[1], alpha=alpha, 
                           **axline_kw)


#############################################
def hex_to_rgb(col):
    """
    hex_to_rgb(col)

    Returns hex color in RGB.

    Required args:
        - col (str): color in hex format

    Returns:
        - col_rgb (list): list of RGB values
    """
    n_comp = 3 # r, g, b
    pos = [1, 3, 5] # start of each component
    leng = 2 

    if "#" not in col:
        raise ValueError("All colors must be provided in hex format.")
    
    # get the int value for each color component
    col_rgb = [int(col[pos[i]:pos[i] + leng], 16) for i in range(n_comp)]

    return col_rgb


#############################################
def rgb_to_hex(col_rgb):
    """
    rgb_to_hex(col_rgb)

    Returns RGB in hex color.

    Required args:
        - col_rgb (list): list of RGB values

    Returns:
        - col (str): color in hex format
    """

    if len(col_rgb) != 3:
        raise ValueError("'col_rgb' must comprise 3 values.")
 
    col = "#{}{}{}".format(*[hex(c)[2:] for c in col_rgb])

    return col


#############################################
def get_col_series(col_ends, n=3):
    """
    get_col_series(col_ends)

    Returns colors between two reference colors, including the two provided.

    Required args:
        - col_ends (list): list of colors in hex format (2)

    Optional args:
        - n (int): number of colors to return, including the 2 provided
                   default: 3

    Returns:
        - cols (list): list of colors between the two reference colors, 
                       including the two provided.
    """

    if len(col_ends) != 2:
        raise ValueError("Must provide exactly 2 reference colours as input.")

    if n < 2:
        raise ValueError("Must request at least 2 colors.")
    else:
        cols = col_ends[:]
        cols_rgb = [hex_to_rgb(col) for col in col_ends]
        div = n - 1
        for i in range(n-2): # for each increment
            vals = []
            for c in range(3): # for each component
                min_val = cols_rgb[0][c]
                max_val = cols_rgb[1][c]
                # get a weighted average for this value
                val = int(
                    np.around((max_val - min_val) * (i + 1)/div + min_val))
                vals.append(val)
            hexval = rgb_to_hex(vals) # add as next to last
            cols.insert(-1, hexval)
    
    return cols


#############################################
def av_cols(cols):
    """
    av_cols(cols)

    Returns average across list of colors provided.

    Required args:
        - cols (list): list of colors in hex format

    Returns:
        - col (str): averaged color in hex format
    """

    cols = gen_util.list_if_not(cols)

    n_comp = 3 # r, g, b
    col_arr = np.empty([len(cols), n_comp])
    for c, col in enumerate(cols):
        col_arr[c] = hex_to_rgb(col)
    col_arr = np.mean(col_arr, axis=0) # average each component
    # extract hex string
    col = rgb_to_hex([int(np.round(c)) for c in col_arr])
    
    return col


#############################################
def incr_ymax(ax, incr=1.1, sharey=False):
    """
    incr_ymax(ax)

    Increases heights of axis subplots.

    Required args:
        - ax (plt Axis): axis

    Optional args:
        - incr (num)   : relative amount to increase subplot height
                         default: 1.1
        - sharey (bool): if True, only the first subplot ymax is modified, as  
                         it will affect all. Otherwise, all subplot ymax are. 
                         default: False
    """

    if sharey:
        change_ax = [get_subax(ax, 0)]
    else:
        n_ax = np.prod(ax.shape)
        change_ax = [get_subax(ax, i) for i in range(n_ax)]
    for sub_ax in change_ax:
        ymin, ymax = sub_ax.get_ylim()
        ymax = (ymax - ymin) * incr + ymin
        sub_ax.set_ylim(ymin, ymax) 


#############################################
def rel_confine_ylims(sub_ax, sub_ran, rel=5):
    """
    rel_confine_ylims(sub_ax, sub_ran)

    Adjusts the y limits of a subplot to confine a specific range to a 
    relative middle range in the y axis. Will not reduce the y lims only
    increase them.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - sub_ran (list)           : range of values corresponding to subrange
                                     [min, max]

    Optional args:
        - rel (num): relative space to be occupied by the specified range, 
                     e.g. 5 for 1/5
                     default: 5
    """

    y_min, y_max = sub_ax.get_ylim()
    sub_min, sub_max = sub_ran 
    sub_cen = np.mean([sub_min, sub_max])

    y_min = np.min([y_min, sub_cen - rel/2 * (sub_cen - sub_min)])
    y_max = np.max([y_max, sub_cen + rel/2 * (sub_max - sub_cen)])

    sub_ax.set_ylim([y_min, y_max])


#############################################
def expand_lims(sub_ax, axis="x", prop=0.2):
    """
    expand_lims(sub_ax)

    Expands the axis limits of a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - axis (str)  : axis for which to expand limits ("x" or "y")
                        default: "x"
        - prop (float): proportion of current axis limits that limits should be 
                        expanded by (total for both ends)
                        default: 0.2
    """

    if axis == "x":
        min_val, max_val = sub_ax.get_xlim()
    elif axis == "y":
        min_val, max_val = sub_ax.get_ylim()
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])
    
    expand = (max_val - min_val) * prop / 2
    new_lims = [min_val - expand, max_val + expand]

    if new_lims[1] < new_lims[0]:
        raise RuntimeError("Expansion requested would flip the axis limits.")

    if axis == "x":
        sub_ax.set_xlim(new_lims)
    elif axis == "y":
        sub_ax.set_ylim(new_lims)


#############################################
def add_vshade(sub_ax, start, end=None, width=None, alpha=0.4, color="k", lw=0, 
               **axspan_kw):
    """
    add_vshade(sub_ax, start)

    Plots shaded vertical areas on subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - start (list)             : list of start position arrays for each 
                                     shaded area (bottom)

    Optional args:
        - end (list)   : list of end position arrays for each shaded area 
                         (takes priority over width)
        - width (num)  : width of the shaded areas
        - alpha (num)  : plt alpha variable controlling shading 
                         transparency (from 0 to 1)
                         default: 0.5
        - color (str)  : color to use
                         default: None
        - lw (num)     : width of the shading edges
                         default: 0

    Kewyord args:
        - axspan_kw (dict): keyword arguments for plt.axvspan() or plt.axhspan()
    """

    start = gen_util.list_if_not(start)
    
    if end is None and width is None:
        raise ValueError("Must specify end or width.")
    elif end is not None:
        end = gen_util.list_if_not(end)
        if len(start) != len(end):
            raise ValueError("end and start must be of the same length.")
        for st, e in zip(start, end):
            sub_ax.axvspan(st, e, alpha=alpha, color=color, lw=lw, 
                           **axspan_kw)
        if width is not None:
            warnings.warn("Cannot specify both end and width. Using end.", 
                category=RuntimeWarning, stacklevel=1)
    else:
        for st in start:
            sub_ax.axvspan(st, st + width, alpha=alpha, color=color, lw=lw, 
                           **axspan_kw)


#############################################
def plot_traces(sub_ax, x, y, err=None, title=None, lw=None, color=None, 
                alpha=0.5, n_xticks=6, xticks=None, yticks=None, label=None, 
                alpha_line=1.0, zorder=None, errx=False, **plot_kw):
    """
    plot_traces(sub_ax, x, y)

    Plots traces (e.g., mean/median with shaded error bars) on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x (array-like)           : array of x values (inferred if None)
        - y (array-like)           : array of y values
        
    Optional args:
        - err (1 or 2D array): either std, SEM or MAD, or quintiles. If 
                               quintiles, 2D array structured as stat x vals
                               default: None
        - title (str)        : subplot title
                               default: None
        - lw (num)           : plt line weight variable
                               default: None
        - color (str)        : color to use
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - n_xticks (int)     : number of xticks (used if xticks is "auto")
                               default: 6
        - xticks (str)       : xtick labels (overrides xticks_ev)
                               ("None" to remove ticks entirely, 
                               "auto" to set xticks automatically from n_xticks)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - label (str)        : label for legend
                               default: None
        - alpha_line (num)   : plt alpha variable controlling line 
                               transparency (from 0 to 1)
                               default: 1.0
        - zorder (int)       : plt zorder variable controlling fore-background 
                               position of line
                               default: None
        - errx (bool)        : if True, error is on the x data, not y data
                               default: False

    Kewyord args:
        - plot_kw (dict): keyword arguments for plt.plot()
    """
    
    if x is None:
        x = range(len(y))

    x = np.asarray(x).squeeze()
    x = x.reshape(1) if len(x.shape) == 0 else x

    y = np.asarray(y).squeeze()
    y = y.reshape(1) if len(y.shape) == 0 else y

    sub_ax.plot(
        x, y, lw=lw, color=color, label=label, alpha=alpha_line, zorder=zorder, 
        **plot_kw)
    color = sub_ax.lines[-1].get_color()
    
    if err is not None:
        err = np.asarray(err).squeeze()
        err = err.reshape(1) if len(err.shape) == 0 else err
        if not errx:
            # only condition where pos and neg error are different
            if len(err.shape) == 2: 
                sub_ax.fill_between(
                    x, err[0], err[1], facecolor=color, alpha=alpha, 
                    zorder=zorder)
            else:
                sub_ax.fill_between(
                    x, y - err, y + err, facecolor=color, alpha=alpha, 
                    zorder=zorder)
        else:
            # only condition where pos and neg error are different
            if len(err.shape) == 2: 
                sub_ax.fill_betweenx(
                    y, err[0], err[1], facecolor=color, alpha=alpha, 
                    zorder=zorder)
            else:
                sub_ax.fill_betweenx(
                    y, x - err, x + err, facecolor=color, alpha=alpha, 
                    zorder=zorder)

    if isinstance(xticks, str): 
        if xticks in ["none", "None"]:
            sub_ax.tick_params(axis="x", which="both", bottom=False) 
        elif xticks == "auto":
            set_ticks_from_vals(sub_ax, x, axis="x", n=n_xticks)
    elif xticks is not None:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)

    if label is not None:
        sub_ax.legend()

    if title is not None:
        sub_ax.set_title(title, y=1.02)


#############################################
def plot_btw_traces(sub_ax, y1, y2, x=None, color="k", alpha=0.5, **fillbtw_kw):
    """
    plot_btw_traces(sub_ax, y1, y2)

    Plots shaded area between x and y lines on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y1 (array-like)          : first array of y values
        - y2 (array-like)          : second array of y values
        
    Optional args:
        - x (array-like)     : array of x values. If None, a range is used.
                               default: None
        - color (str)        : color to use. If a list is provided, the
                               average is used.
                               default: "k"
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5

    Kewyord args:
        - fillbtw_kw (dict): keyword arguments for plt.fill_between()
    """

    y1 = np.asarray(y1).squeeze()
    y1 = y1.reshape(1) if len(y1.shape) == 0 else y1

    y2 = np.asarray(y2).squeeze()
    y2 = y2.reshape(1) if len(y2.shape) == 0 else y2

    if x is None:
        x = list(range(len(y1)))
    else:
        x = np.asarray(x).squeeze()
        x = x.reshape(1) if len(x.shape) == 0 else x

    if len(y1) != len(y2) or len(x) != len(y1):
        raise ValueError("y1 and y2, and x if provided, must have the same "
            "length.")

    comp_arr = np.concatenate([y1[:, np.newaxis], y2[:, np.newaxis]], axis=1)
    maxes = np.max(comp_arr, axis=1)
    mins  = np.min(comp_arr, axis=1)

    if isinstance(color, list):
        color = av_cols(color)

    sub_ax.fill_between(x, mins, maxes, alpha=alpha, facecolor=color, 
                        **fillbtw_kw)


#############################################
def plot_errorbars(sub_ax, y, err=None, x=None, title=None, alpha=0.8, 
                   xticks=None, yticks=None, label=None, fmt="-o", 
                   line_dash=None, **errorbar_kw):
    """
    plot_errorbars(sub_ax, y)

    Plots points with errorbars on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y (array-like)           : array of y values

    Optional args:
        - err (1 or 2D array): either std, SEM or MAD, or quintiles. If 
                               quintiles, 2D array structured as stat x vals
                               default: None
        - x (array-like)     : array of x values. 
                               default: None
        - title (str)        : subplot title
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - xticks (str)       : xtick labels ("None" to remove ticks entirely, 
                               "auto" to set xticks automatically)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - label (str)        : label for legend
                               default: None
        - fmt (str)          : data point/lines format.
                               default: "-o"
        - line_dash (str)    : dash pattern for the data line 
                               (not the errorbar components)
                               default: None

    Kewyord args:
        - errorbar_kw (dict): keyword arguments for plt.errorbar(), 
                              e.g. linewidth, color, markersize, capsize
    """
    
    y = np.asarray(y).squeeze()
    y = y.reshape(1) if len(y.shape) == 0 else y
    
    if x is None:
        x = np.arange(1, len(y) + 1)
    
    if isinstance(xticks, str):
        if xticks in ["None", "none"]:
            sub_ax.tick_params(axis="x", which="both", bottom=False) 
        elif xticks == "auto":
            set_ticks_safe(sub_ax, x, axis="x")
    elif xticks is not None:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)

    x = np.asarray(x).squeeze()
    x = x.reshape(1) if len(x.shape) == 0 else x

    if err is not None:
        err = np.asarray(err).squeeze()
        err = err.reshape(1) if len(err.shape) == 0 else err
        # If err is 1D, errorbar length is provided, if err is 2D, errorbar 
        # endpoints are provided
        if len(err.shape) == 2: 
            err = np.asarray([y - err[0], err[1] - y])
    
    if err is not None and not np.sum(np.isfinite(err)):
        err = None 

    sub_ax.errorbar(x, y, err, label=label, fmt=fmt, alpha=alpha, 
                    **errorbar_kw)

    if line_dash is not None:
        sub_ax.get_lines()[-3].set_linestyle(line_dash)

    if label is not None:
        sub_ax.legend()

    if title is not None:
        sub_ax.set_title(title, y=1.02)


#############################################
def plot_two_color_errorbars(sub_ax, y, mask, colors, err=None, x=None, 
                             link_left=True, title=None, alphas=0.8, 
                             xticks=None, yticks=None, labels=None, fmt="-o", 
                             **errorbar_kw):
    """
    plot_two_color_errorbars(sub_ax, y, mask, colors)

    Plots points with errorbars on subplot (ax) in two sets of colors.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y (array-like)           : array of y values
        - mask (array-like)        : mask for first color values
        - colors (list)            : colors for [True, False] values of mask

    Optional args:
        - err (1 or 2D array) : either std, SEM or MAD, or quintiles. If 
                                quintiles, 2D array structured as stat x vals
                                default: None
        - x (array-like)      : array of x values. 
                                default: None
        - link_left (bool)    : if True, markers share a color with the line 
                                leaving them to the left. If False, it is to 
                                the right.
                                default: True
        - title (str)         : subplot title
                                default: None
        - alphas (num or list): plt alpha variable controlling shading 
                                transparency (from 0 to 1)
                                default: 0.5
        - xticks (str)        : xtick labels ("None" to remove ticks entirely, 
                                "auto" to set xticks automatically)
                                default: None
        - yticks (str)        : ytick labels
                                default: None
        - labels (list)       : labels for each color
                                default: None
        - fmt (str)           : data point/lines format.
                                default: "-o"

    Kewyord args:
        - errorbar_kw (dict): keyword arguments for plt.errorbar(), 
                              e.g. linewidth, color, markersize, capsize
    """

    y = np.asarray(y).squeeze()
    y = y.reshape(1) if len(y.shape) == 0 else y
    
    if x is None:
        x = np.arange(1, len(y) + 1)

    x = np.asarray(x).squeeze()
    x = x.reshape(1) if len(x.shape) == 0 else x

    mask = np.asarray(mask).squeeze()
    mask = mask.reshape(1) if len(mask.shape) == 0 else mask

    if len(mask) != len(x) or len(x) != len(y):
        raise ValueError("y, mask and x must have the same length.")

    if labels is None:
        labels = [None, None]
    if len(colors) != 2 or len(labels) != 2:
        raise ValueError("Must provide 2 colors, and 2 labels, if any.")
    if isinstance(alphas, list):
        if len(alphas) != 2:
            raise ValueError("Must provide 2 alphas, if providing more than 1.")
    else:
        alphas = [alphas] * 2

    # reverse if needed
    if link_left:
        x = x[::-1]
        y = y[::-1]
        mask = mask[::-1]

    # add linked lines
    for mask_targ, color, alpha in zip([True, False], colors, alphas):
        # implemented to link right
        col_x, col_y = [], []
        for m, mask_val in enumerate(mask == mask_targ):
            if mask_val or m == len(mask) - 1:
                col_x.append(x[m])
                col_y.append(y[m])
            else:
                col_x.extend([x[m], np.mean(x[m : m+1])])
                col_y.extend([y[m], np.nan])
        
        if link_left:
            col_x = col_x[::-1]
            col_y = col_y[::-1]

        plot_errorbars(sub_ax, col_y, err=None, x=col_x, alpha=alpha, 
            fmt=fmt, ms=0, color=color)

    # unreverse
    if link_left:
        x = x[::-1]
        y = y[::-1]
        mask = mask[::-1]

    # add lone markers (with errorbars)
    if err is not None:
        err = np.asarray(err).squeeze()
        err = err.reshape(1) if len(err.shape) == 0 else err
        if not np.sum(np.isfinite(err)):
            err = None
        elif err.shape[-1] != len(mask):
            raise ValueError("err must have the same length as the mask.")
    
    fmt_marker = fmt.replace("-", "")
    for mask_targ, color, label, alpha in zip(
        [True, False], colors, labels, alphas
        ):
        use_mask = (mask == mask_targ)
        use_err = err[..., use_mask] if err is not None else None
        plot_errorbars(sub_ax, y[use_mask], err=use_err, x=x[use_mask], 
            title=title, alpha=alpha, xticks=xticks, yticks=yticks, 
            fmt=fmt_marker, color=color, label=label, **errorbar_kw)
            
            
#############################################
def get_barplot_xpos(n_grps, n_bars_per, barw, in_grp=1.5, btw_grps=4.0):
    """
    get_barplot_xpos(n_grps, n_bars_per, barw)

    Returns center positions, bar positions and x limits to position bars in a 
    barplot in dense groups along the axis. 

    Required args:
        - n_grps (int)    : number of groups along the x axis
        - n_bars_per (int): number of bars within each group
        - barw (num)      : width of each bar

    Optional args:
        - in_grp (num)  : space between bars in a group, relative to bar 
                          default: 1.5
        - btw_grps (num): space between groups, relative to bar
                          (also determines space on each end, which is half)
                          default: 4.0

    Returns:
        - center_pos (list)    : central position of each group
        - bar_pos (nested list): position of each bar, structured as:
                                    grp x bar
        - xlims (list)         : x axis limit range
    """

    in_grp   = float(in_grp)
    btw_grps = float(btw_grps)
    barw     = float(barw)
    
    # space for each group, relative to barw
    per_grp = n_bars_per + in_grp * (n_bars_per - 1)
    
    # center position of each group
    center_pos = [barw * (x + .5) * (per_grp + btw_grps) 
                                  for x in range(n_grps)]

    # bar positions
    center_idx = (n_bars_per - 1)/2.
    btw_bars = (1 + in_grp) * barw # between bar centers
    bar_pos = [[pos + (i - center_idx) * btw_bars for i in range(n_bars_per)] 
        for pos in center_pos]

    xlims = [0, n_grps * barw * (per_grp + btw_grps)]

    return center_pos, bar_pos, xlims


#############################################
def add_signif_mark(sub_ax, xpos, yval, yerr=None, rel_y=0.01, color="k", 
                    fig_coord=False, fontsize="xx-large", fontweight="bold", 
                    ha="center", va="top", mark="*", **text_kw):
    """
    add_signif_mark(sub_ax, xpos, yval)

    Plots significance markers (mark) on subplot.

    Best to ensure that y axis limits are set before calling this function as
    mark position are set relative to these limits.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - xpos (num)               : x positions for mark
        - yval (num)               : y value above which to place line
    
    Optional args:
        - yerr (num)      : errors to add to ypos when placing mark
                            default: None
        - rel_y (num)     : relative position above ypos at which to place mark.
                            default: 0.01
        - color (str)     : color for marks
                             default: "k"
        - fig_coord (bool): if True, coordinates are first converted from data 
                            to figure coordinates
                            default: False
        - fontsize (str)  : text (mark) fontsize
                            default: "xx-large"
        - fontweight (str): text (mark) fontweight
                            default: "bold"
        - ha (str)        : text (mark) horizontal alignment
                            default: "center" 
        - va (str)        : text (mark) vertical alignment
                            default: "top"
        - mark (str)      : mark to use to mark significance
                            default: "*"

    Kewyord args:
        - text_kw (dict): keyword arguments for plt.text()
   """

    rel_y = float(rel_y)
    
    # y positions
    if yerr is not None:
        yval = yval + yerr

    ylims = sub_ax.get_ylim()

    # y text position (will appear higher than line)
    ytext = yval + (rel_y * (ylims[1] - ylims[0]))

    obj = sub_ax
    if fig_coord:
        obj = sub_ax.figure
        xpos, ytext = obj.transFigure.inverted().transform(
            sub_ax.transData.transform([xpos, ytext]))

    obj.text(xpos, ytext, mark, color=color, fontsize=fontsize, 
        fontweight=fontweight, ha=ha, va=va, **text_kw)


#############################################
def plot_barplot_signif(sub_ax, xpos, yval, yerr=None, rel_y=0.02, color="k", 
                        lw=2, mark_rel_y=0.05, mark="*", **text_kw):
    """
    plot_barplot_signif(sub_ax, xpos, yval)

    Plots significance markers (line and mark) above bars showing a significant
    difference. 
    Best to ensure that y axis limits are set before calling this function as
    line and mark position are set relative to these limits.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - xpos (array-like)        : list of x positions for line to span
        - ypos (array-like)        : list of y values above which to place line
    
    Optional args:
        - yerr (array-like): list of errors to add to ypos when placing line
                             default: None
        - rel_y (num)      : relative position above ypos at which to place
                             line.
                             default: 0.02
        - color (str)      : line and mark colour
                             default: "k"
        - lw (int)         : line width
                             default: 2
        - mark_rel_y (num) : relative position above bar at which to place mark
                             default: 0.02
        - mark (str)       : significance marker
                             default: "*"

    Kewyord args:
        - text_kw (dict): keyword arguments for plt.text()
    """

    rel_y = float(rel_y)
    
    # x positions
    if len(xpos) < 2:
        raise ValueError("xpos must be at least of length 2.")
    xpos = [np.min(xpos), np.max(xpos)]
    xmid = np.mean(xpos)

    # y positions
    yval = np.asarray(yval)
    if not yval.shape:
        yval = yval.reshape(-1)
    if yerr is None:
        yerr = np.zeros_like(yval)

    for y, err in enumerate(yerr): # in case of NaNs
        if np.isnan(err):
            yerr[y] = 0

    if len(yval) != len(yerr):
        raise ValueError("If provided, yerr must have the same length as yval.")

    # if quintiles are provided, the second (high) one is retained
    if yerr.shape == 2:
        yerr = yerr[:, 1]

    ymax = np.max(yval + yerr)
    ylims = sub_ax.get_ylim()

    # y line position
    yline = ymax + rel_y * (ylims[1] - ylims[0])
    
    # place y text slightly higher
    add_signif_mark(sub_ax, xmid, ymax, rel_y=mark_rel_y, color=color, 
        mark=mark, **text_kw)

    sub_ax.plot(xpos, [yline, yline], lw=lw, color=color)


#############################################
def plot_bars(sub_ax, x, y, err=None, title=None, width=0.75, lw=None, 
              alpha=0.5, xticks=None, yticks=None, xlims=None, label=None, 
              hline=None, capsize=8, **bar_kw):
    """
    plot_bars(sub_ax, chunk_val)

    Plots bars (e.g., mean/median with shaded error bars) on subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x (array-like)           : list of x values

    Optional args:
        - err (1 or 2D array): either std, SEM or MAD, or quintiles. If 
                               quintiles, 2D array structured as stat x vals
                               default: None
        - title (str)        : subplot title
                               default: None
        - lw (num)           : plt line weight variable
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - xticks (str)       : xtick labels ("None" to remove ticks entirely, 
                               "auto" to set xticks automatically)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - xlims (list)       : xlims
                               default: None
        - label (str)        : label for legend
                               default: None
        - hline (list or num): list of y coordinates at which to add 
                               horizontal bars
                               default: None
        - capsize (num)      : length of errorbar caps
                               default: 8

    Kewyord args:
        - bar_kw (dict): keyword arguments for plt.bar()
    """
    
    x = np.asarray(x).squeeze()
    x = x.reshape(1) if len(x.shape) == 0 else x

    y = y.squeeze()
    y = y.reshape(1) if len(y.shape) == 0 else y

    patches = sub_ax.bar(x, y, width=width, lw=lw, label=label, **bar_kw)
    
    # get color
    fc = patches[0].get_fc()

    # add errorbars
    if err is not None and np.sum(np.isfinite(err)):
        plot_errorbars(sub_ax, y, err=err, x=x, fmt="None", elinewidth=lw, 
            capsize=capsize, capthick=lw, ecolor=fc)

    # set edge color to match patch face color
    [patch.set_ec(fc) for patch in patches]

    # set face color to transparency
    [patch.set_fc(list(fc[0:3]) + [alpha]) for patch in patches]

    if label is not None:
        sub_ax.legend()

    if xlims is not None:
        sub_ax.set_xlim(xlims)
    
    if hline is not None:
        sub_ax.axhline(y=hline, c="k", lw=1.5)
    
    if isinstance(xticks, str):
        if xticks in ["None", "none"]:
            sub_ax.tick_params(axis="x", which="both", bottom=False) 
        elif xticks == "auto":
            set_ticks_safe(sub_ax, x, axis="x")
    elif xticks is not None:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)
    
    if title is not None:
        sub_ax.set_title(title, y=1.02)


#############################################
def add_colorbar(fig, im, n_cols, label=None, cm_prop=0.03, space_fact=2, 
                 **cbar_kw):
    """
    add_colorbar(fig, im, n_cols)

    Adds a slim colorbar to the right side of a figure.

    Required args:
        - fig (plt Fig)     : figure
        - n_cols (int)      : number of columns in figure
        - im (plt Colormesh): colormesh

    Optional args:
        - label (str)     : colormap label
                            default: None
        - cm_prop (float) : colormap width wrt figure size, to be scaled by 
                            number of columns
                            default: 0.03
        - space_fact (num): factor by which to extend figure spacing, 
                            proportionally to cm_prop
                            default: 2
    
    Kewyord args:
        - cbar_kw (dict): keyword arguments for plt.colorbar()

    Returns:
        - cbar (plt Colorbar): pyplot colorbar
    """

    cm_w = cm_prop / n_cols
    fig.subplots_adjust(right=1 - cm_w * space_fact)
    cbar_ax = fig.add_axes([1, 0.15, cm_w, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, **cbar_kw)

    if label is not None:
        cbar.set_label(label)

    return cbar


#############################################
def plot_colormap(sub_ax, data, xran=None, yran=None, title=None, cmap=None, 
                  n_xticks=6, xticks=None, yticks_ev=10, xlims=None, 
                  ylims=None, origin=None, **cmesh_kw):
    """
    plot_colormap(sub_ax, data)

    Plots colormap on subplot and returns colormesh image.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - data (2D array)          : data array (X x Y)

    Optional args:
        - xran (list)    : first and last values along the x axis. If None,
                           will be inferred from the data.
                           default: None
        - yran (list)    : first and last values along the y axis. If None,
                           will be inferred from the data.
                           default: None
        - title (str)    : subplot title
                           default: None
        - cmap (colormap): a matplotlib colormap
                           default: None
        - n_xticks (int) : number of xtick labels (used if xticks is "auto")
                           default: 6
        - xticks (str)   : xtick labels (overrides n_xticks)
                           ("auto" to set xticks automatically with n_xticks)
                           default: None
        - yticks_ev (str): frequency at which to set ytick labels
                           default: None
        - xlims (list)   : xlims
                           default: None
        - ylims (list)   : ylims
                           default: None
        - origin (str)   : where to position the y axis origin 
                           ("upper" or "lower")
                           default: "upper"
    
    Kewyord args:
        - cmesh_kw (dict): keyword arguments for plt.pcolormesh()

    Returns:
        - im (plt Colormesh): colormesh image
    """
    
    if xran is None:
        xran = np.linspace(0.5, data.shape[0]+0.5, data.shape[0]+1)
    else:
        xran = np.linspace(xran[0], xran[1], data.shape[0]+1)

    if yran is None:
        yran = np.linspace(0.5, data.shape[1]+0.5, data.shape[1]+1)
    else:
        yran = np.linspace(yran[0], yran[1], data.shape[1]+1)

    if yticks_ev is not None:
        yticks = list(range(0, data.shape[1], yticks_ev))
        sub_ax.set_yticks(yticks)
    
    if isinstance(xticks, str) and xticks == "auto":
        xticks = np.linspace(np.min(xran), np.max(xran), n_xticks)
    if xticks is not None:
        sub_ax.set_xticks(xticks)

    im = sub_ax.pcolormesh(xran, yran, data.T, cmap=cmap, **cmesh_kw)
    
    if xlims is not None:
        sub_ax.set_xlim(xlims)

    if ylims is not None:
        sub_ax.set_ylim(ylims)

    if title is not None:
        sub_ax.set_title(title, y=1.02)
    
    # check whether y axis needs to be flipped, based on origin
    if origin is not None:
        ylims = sub_ax.get_ylim()
        flip = False
        if (origin == "lower") and (ylims[1] < ylims[0]):
            flip = True
        elif (origin == "upper") and (ylims[1] > ylims[0]):
            flip = True
        elif origin not in ["lower", "upper"]:
            gen_util.accepted_values_error("origin", origin, ["lower", "upper"])
        if flip:
            sub_ax.set_ylim((ylims[1], ylims[0]))

    return im


#############################################
def plot_sep_data(sub_ax, data, lw=0.1, no_edges=True):
    """
    plot_sep_data(sub_ax, data)

    Plots data separated along the first axis, so that each item is scaled to 
    within a unit range, and shifted by 1 from the previous item. Allows 
    items in the same range to be plotted in a stacked fashion. 

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - data (2D array)          : data array (items x values)

    Optional args:
        - lw (num)       : linewidth
        - no_edges (bool): if True, the edges and ticks are removed and the 
                           y limits are tightened
                           default: True
    """

    data_sc = math_util.scale_data(data, axis=1, sc_type="min_max")[0]
    add = np.linspace(0, data.shape[0] * 1.2 + 1, data.shape[0])[:, np.newaxis]
    data_sep = data_sc + add

    sub_ax.plot(data_sep.T, lw=lw)

    if no_edges:
        # removes ticks
        remove_ticks(sub_ax, True, True)
        # removes subplot edges
        remove_graph_bars(sub_ax, bars="all")
        # tighten y limits
        sub_ax.set_ylim(np.min(data_sep), np.max(data_sep))


#############################################
def plot_lines(sub_ax, y, x=None, y_rat=0.0075, color="black", width=0.4, 
               alpha=1.0, **bar_kw):
    """
    plot_lines(sub_ax, y)

    Plots lines for each x value at specified height on a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y (array-like)           : array of y values for each line

    Optional args:
        - x (array-like): array of x values
                          default: None
        - y_rat (float) : med line thickness, relative to subplot height
                          default: 0.0075
        - col0r (str)   : bar color
                          default: "black"
        - width (float) : bar thickness
                          default: 0.4
        - alpha (num)   : plt alpha variable controlling shading 
                          transparency (from 0 to 1)
                          default: 0.5

    Kewyord args:
        - bar_kw (dict): keyword arguments for plt.bar()
    """

    if x is None:
        x = range(len(y))
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same last length.")

    y_lim = sub_ax.get_ylim()
    y_th = y_rat * (y_lim[1] - y_lim[0])
    bottom = y - y_th/2.
    sub_ax.bar(
        x, height=y_th, bottom=bottom, color=color, width=width, alpha=alpha, 
        **bar_kw)


#############################################
def plot_ufo(sub_ax, x, y, err=None, color="k", width=0.5, thickness=4.5, 
             no_line=False, **errorbar_kw):
    """
    plot_ufo(sub_ax, x, y)

    Plots data as a circle with a line behind it.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x (array-like)           : x values
        - y (array-like)           : y values
    
    Optional args:
        - err (num or array-like): either std, SEM or MAD, or quintiles
                                   default: None
        - color (str)            : color
                                   default: "k"
        - no_line (bool)         : if True, line is omitted
                                   default: False
    """

    if not no_line:
        # plot actual data as line
        x_edges = np.asarray([
            [i - width / 2, i + width / 2] for i in np.asarray(x).reshape(-1)
            ]).T
        y_edges = np.tile(y, 2).reshape(2, -1)
        sub_ax.plot(x_edges, y_edges, lw=thickness, color=color)

    # plot errorbars
    if err is not None:
        if not isinstance(err, (list, np.ndarray)):
            err = [err]
        plot_errorbars(sub_ax, y, err=err, x=x, color=color, alpha=0.8, 
            **errorbar_kw)

    if not no_line:
        # plot a white circle center
        sub_ax.plot(x, y, color="white", marker="o", markersize=13, 
                    markeredgewidth=2.5)

    # plot circle edge
    sub_ax.plot(x, y, color=color, marker="o", markersize=13, 
                fillstyle="none", markeredgewidth=2.5)
    # plot circle center
    sub_ax.plot(x, y, color=color, marker="o", markersize=13, alpha=0.5)


#############################################
def plot_CI(sub_ax, extr, med=None, x=None, width=0.4, label=None, 
            color="lightgray", med_col="gray", med_rat=0.015, zorder=None):
    """
    plot_CI(sub_ax, extr)

    Plots confidence intervals on a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - extr (2D array-like)     : array of CI extrema, structured 
                                     as perc [low, high] x bar
    Optional args:
        - med (array-like): array of median/mean values for each bar. If 
                            None, no median line is added
                            default: None
        - x (array-like)  : array of x values (if None, they are inferred)
                            default: None
        - width (float)   : bar thickness
                            default: 0.4
        - label (str)     : label for the bars
                            default: None
        - color (str)     : bar color
                            default: "lightgray"
        - med_col (str)   : med line color, if med is provided
                            default: "gray"
        - med_rat (float) : med line thickness, relative to subplot height
                            default: 0.015
        - zorder (int)    : plt zorder variable controlling fore-background 
                            position of line/shading
                            default: None
    """

    x = np.asarray(x).reshape(-1)
    med = np.asarray(med).reshape(-1)

    extr = np.asarray(extr)
    if len(extr.shape) == 1:
        extr = extr.reshape([-1, 1])
    if extr.shape[0] != 2:
        raise ValueError("Must provide exactly 2 extrema values for each bar.")

    if x is None:
        x = range(len(extr.shape[1]))
    if len(x) != extr.shape[1]:
        raise ValueError("'x' and 'extr' must have the same last "
            "dimension length.")

    # plot CI
    sub_ax.bar(x, height=extr[1]-extr[0], bottom=extr[0], color=color, 
               width=width, label=label, zorder=zorder)
    
    if label is not None:
        sub_ax.legend()

    # plot median (with some thickness based on ylim)
    if med is not None:
        med = np.asarray(med)
        if len(x) != len(med):
            raise ValueError("'x' and 'med' must have the same last "
                "dimension length.")
        
        plot_lines(
            sub_ax, med, x, med_rat, color=med_col, width=width, 
            zorder=zorder)


#############################################
def plot_data_cloud(sub_ax, x_val, y_vals, disp_wid=0.3, label=None, 
                    color="k", alpha=0.5, randst=None, **plot_kw):
    """
    plot_data_cloud(sub_ax, x_val, y_vals)

    Plots y values as a data cloud around an x value

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x_val (float)            : center of data
        - y_vals (array-like)      : array of y values for each marker.
                                     default: None

    Optional args:
        - disp_std (float)   : dispersion standard deviation 
                               (will clip at 2.5 * disp_std)
                               default: 0.4
        - label (str)        : label for the bars
                               default: None
        - color (str)        : marker color
                               default: "k"
        - alpha (float)      : transparency
                               default: 0.5

    Kewyord args:
        - plot_kw (dict): keyword arguments for plt.plot()
    
    Returns:
        - cloud (plt Line): pyplot Line object containing plotted dots
    """

    randst = rand_util.get_np_rand_state(randst)

    x_vals = randst.normal(x_val, disp_wid, len(y_vals))

    # clip points outside 2.5 stdev
    min_val, max_val = [x_val + sign * 2.5 * disp_wid for sign in [-1, 1]]
    x_vals[np.where(x_vals < min_val)] = min_val
    x_vals[np.where(x_vals > max_val)] = max_val

    cloud = sub_ax.plot(
        x_vals, y_vals, marker=".", lw=0, color=color, alpha=alpha, 
        label=label, **plot_kw)[0]

    if label is not None:
        sub_ax.legend()

    return cloud

    
#############################################
def get_fig_rel_pos(ax, grp_len, axis="x"):
    """
    get_fig_rel_pos(ax, grp_len)

    Gets figure positions for middle of each subplot grouping in figure 
    coordinates.

    Required args:
        - ax (plt Axis): axis
        - grp_len (n)  : grouping

    Optional args:
        - axis (str): axis for which to get position ("x" or "y")
                      default: "x"
    Returns:
        - poses (list): positions for each group
    """


    if not isinstance(ax, np.ndarray) and len(ax.shape):
        raise ValueError("ax must be a 2D numpy array.")

    fig = ax.reshape(-1)[0].figure
    n_rows, n_cols = ax.shape
    poses = []
    if axis == "x":
        if n_cols % grp_len != 0:
            raise RuntimeError(f"Group length of {grp_len} does not fit with "
                f"{n_cols} columns.")
        n_grps = int(n_cols/grp_len)
        for n in range(n_grps):
            left_subax = ax[0, n * grp_len]
            left_pos = fig.transFigure.inverted().transform(
                left_subax.transAxes.transform([0, 0]))[0]

            right_subax = ax[0, (n + 1) * grp_len - 1]
            right_pos = fig.transFigure.inverted().transform(
                right_subax.transAxes.transform([1, 0]))[0]

            poses.append(np.mean([left_pos, right_pos]))
    elif axis == "y":
        if n_rows % grp_len != 0:
            raise RuntimeError(f"Group length of {grp_len} does not fit with "
                f"{n_rows} rows.")
        n_grps = int(n_rows/grp_len)
        for n in range(n_grps):
            top_subax = ax[n * grp_len, 0]
            top_pos = fig.transFigure.inverted().transform(
                top_subax.transAxes.transform([0, 1]))[1]

            bottom_subax = ax[(n + 1) * grp_len - 1, 0]
            bottom_pos = fig.transFigure.inverted().transform(
                bottom_subax.transAxes.transform([0, 0]))[1]

            poses.append(np.mean([top_pos, bottom_pos]))
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])

    return poses


#############################################
def set_interm_ticks(ax, n_ticks, axis="x", share=True, skip=True, 
                     update_ticks=False, **font_kw):
    """
    set_interm_ticks(ax, n_ticks)

    Sets axis tick values based on number of ticks, either all as major ticks, 
    or as major ticks with skipped, unlabelled ticks in between. When possible, 
    0 and top tick are set as major ticks.

    Required args:
        - ax (plt Axis): axis
        - n_ticks (n)  : max number of labelled ticks

    Optional args:
        - axis (str)         : axis for which to set ticks ("x" or "y")
                               default: "x"
        - share (bool)       : if True, all axes set the same, based on first 
                               axis.
                               default: True
        - skip (bool)        : if True, intermediate ticks are unlabelled. If 
                               False, all ticks are labelled
                               default: True
        - update_ticks (bool): if True, ticks are updated to axis limits first
                               default: False

    Kewyord args:
        - font_kw (dict): keyword arguments for plt.yticklabels() or 
                          plt.xticklabels() fontdict, e.g. weight
    """

    if not isinstance(ax, np.ndarray):
        raise TypeError("Must pass an axis array.")
    
    if n_ticks < 2:
        raise ValueError("n_ticks must be at least 2.")

    for s, sub_ax in enumerate(ax.reshape(-1)):
        if s == 0 or not share:
            if axis == "x":
                if update_ticks:
                    sub_ax.set_xticks(sub_ax.get_xlim())
                ticks = sub_ax.get_xticks()
            elif axis == "y":
                if update_ticks:
                    sub_ax.set_yticks(sub_ax.get_ylim())
                ticks = sub_ax.get_yticks()
            else:
                gen_util.accepted_values_error("axis", axis, ["x", "y"])

            diff = np.mean(np.diff(ticks)) # get original tick steps
            if len(ticks) >= n_ticks:
                ratio = np.ceil(len(ticks) / n_ticks)
            else:
                ratio = 1 / np.ceil(n_ticks / len(ticks))
    
            step = diff * ratio / 2

              # 1 signif digit for differences
            if step == 0:
                o = 0
            else:
                o = -int(math_util.get_order_of_mag(step * 2))

            step = np.around(step * 2, o) / 2
            step = step * 2 if not skip else step

            min_tick_idx = np.round(np.min(ticks) / step).astype(int)
            max_tick_idx = np.round(np.max(ticks) / step).astype(int)

            tick_vals = np.linspace(
                min_tick_idx * step, 
                max_tick_idx * step, 
                max_tick_idx - min_tick_idx + 1
                )

            idx = np.where(tick_vals == 0)[0]
            if 0 not in tick_vals:
                idx = 0

            # adjust if only 1 tick is labelled
            if skip and (len(tick_vals) < (3 + idx % 2)):
                max_tick_idx += 1
                tick_vals = np.append(tick_vals, tick_vals[-1] + step)

            labels = []
            final_tick_vals = []
            for v, val in enumerate(tick_vals):
                val = np.around(val, o + 3) # to avoid floating point precision problems
                final_tick_vals.append(val)                    
                
                if (v % 2 == idx % 2) or not skip:
                    val = int(val) if int(val) == val else val
                    labels.append(val)
                else:
                    labels.append("")

        if axis == "x":
            sub_ax.set_xticks(final_tick_vals)
            # always set ticks (even again) before setting labels
            sub_ax.set_xticklabels(labels, fontdict=font_kw)
            # adjust limits if needed
            lims = list(sub_ax.get_xlim())
            if final_tick_vals[-1] > lims[1]:
                lims[1] = final_tick_vals[-1]
            if final_tick_vals[0] < lims[0]:
                lims[0] = final_tick_vals[0]
            sub_ax.set_xlim(lims)
    
        elif axis == "y":
            sub_ax.set_yticks(final_tick_vals)
            # always set ticks (even again) before setting labels
            sub_ax.set_yticklabels(labels, fontdict=font_kw)
            # adjust limits if needed
            lims = list(sub_ax.get_ylim())
            if final_tick_vals[-1] > lims[1]:
                lims[1] = final_tick_vals[-1]
            if final_tick_vals[0] < lims[0]:
                lims[0] = final_tick_vals[0]
            sub_ax.set_ylim(lims)


#############################################
def set_minimal_ticks(sub_ax, axis="x", **font_kw):
    """
    set_minimal_ticks(sub_ax)

    Sets minimal ticks for a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - axis (str): axes for which to set ticks ("x" or "y")
                      default: "x"

    Kewyord args:
        - font_kw (dict): keyword arguments for plt.yticklabels() or 
                          plt.xticklabels() fontdict, e.g. weight
    """

    sub_ax.autoscale()

    if axis == "x":
        lims = sub_ax.get_xlim()
    elif axis == "y":
        lims = sub_ax.get_ylim()
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])

    ticks = rounded_lims(lims)

    if np.sign(ticks[0]) != np.sign(ticks[1]):
        if np.absolute(ticks[1]) > np.absolute(ticks[0]):
            ticks = [0, ticks[1]]
        elif np.absolute(ticks[1]) < np.absolute(ticks[0]):
            ticks = [ticks[0], 0]
        else:
            ticks = [ticks[0], 0, ticks[1]]

    if axis == "x":
        sub_ax.set_xticks(ticks)
        sub_ax.set_xticklabels(ticks, fontdict=font_kw)
    elif axis == "y":
        sub_ax.set_yticks(ticks)
        sub_ax.set_yticklabels(ticks, fontdict=font_kw)
    


#############################################
def rounded_lims(lims, out=False):
    """
    rounded_lims(lims)

    Returns axis limit values rounded to the nearest order of magnitude.

    Required args:
        - lims (iterable): axis limits (lower, upper)

    Optional args:
        - out (bool): if True, limits are only ever rounded out.
                      default: False

    Returns:
        - new_lims (list): rounded axis limits [lower, upper]
    """

    new_lims = list(lims)[:]
    lim_diff = lims[1] - lims[0]

    if lim_diff != 0:
        order = math_util.get_order_of_mag(lim_diff)
        o = -int(order) 

        new_lims = []
        for l, lim in enumerate(lims):
            round_fct = np.around
    
            if lim < 0:
                if out:
                    round_fct = np.ceil if l == 0 else np.floor
                new_lim = -round_fct(-lim * 10 ** o)
            else:
                if out:
                    round_fct = np.floor if l == 0 else np.ceil
                new_lim = round_fct(lim * 10 ** o)

            new_lim = new_lim / 10 ** o
            
            new_lims.append(new_lim)

    return new_lims


#############################################
def adjust_tick_labels_for_sharing(axis_set, axes="x"):
    """
    adjust_tick_labels_for_sharing(axis_set)

    Adjust presence of axis ticks labels for sharing. 

    Required args:
        - axis_set (list): axes to group

    Optional args:
        - axes (str or list): axes ("x", "y") to group
    """
    
    axes = gen_util.list_if_not(axes)
    for axis in axes:
        if axis == "x":
            row_ns = [subax.get_subplotspec().rowspan.start 
                for subax in axis_set]
            last_row_n = np.max(row_ns)

            for subax in axis_set:
                if subax.get_subplotspec().rowspan.start != last_row_n:
                    subax.tick_params(axis="x", labelbottom=False)

        elif axis == "y":
            col_ns = [subax.get_subplotspec().colspan.start 
                for subax in axis_set]
            first_col_n = np.min(col_ns)

            for subax in axis_set:
                if subax.get_subplotspec().colspan.start != first_col_n:
                    subax.tick_params(axis="y", labelleft=False)
        
        else:
            gen_util.accepted_values_error("axis", axis, ["x", "y"])


#############################################
def get_shared_axes(ax, axis="x"):
    """
    get_shared_axes(ax)

    Returns lists of subplots that share an axis, compensating for what appears 
    to be a bug in matplotlib where subplots from different figures accumulate 
    at each call.

    Required args:
        - ax (plt Axis): axis

    Optional args:
        - axis (str): axis for which to get grouping

    Returns:
        - fixed_grps (list): subplots, organized by group that share the axis
    """


    all_subplots = ax.reshape(-1).tolist()

    if axis == "x":
        grps = list(all_subplots[0].get_shared_x_axes())
    elif axis == "y":
        grps = list(all_subplots[0].get_shared_y_axes())
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])

    fixed_grps = []
    for grp in grps:
        fixed_grp = []
        for subplot in grp:
            if subplot in all_subplots:
                fixed_grp.append(subplot)
        if len(fixed_grp) != 0:
            fixed_grps.append(fixed_grp)

    return fixed_grps


#############################################
def set_shared_axes(axis_set, axes="x", adjust_tick_labels=False):
    """
    set_shared_axes(ax)

    Sets axis set passed to be shared. 
    
    Not sure how this interacts with the matplotlib bug described in 
    get_shared_axes() above. Alternative methods didn't work though. Bugs may 
    arise in the future when multiple figures are opened consecutively.

    Relevant matplotlib source code:
        - matplotlib.axes._base: where get_shared_x_axes() is defined
        - matplotlib.cbook.Grouper: where the grouper class is defined

    Required args:
        - axis_set (list): axes to group

    Optional args:
        - axes (str or list)       : axes ("x", "y") to group
                                     default: "x"
        - adjust_tick_labels (bool): if True, tick labels are adjusted for axis 
                                     sharing. (Otherwise, only the limits are 
                                     shared, but tick labels are repeated.)
                                     default: False
    """

    axes = gen_util.list_if_not(axes)
    for axis in axes:
        if axis == "x":
            grper = axis_set[0].get_shared_x_axes()
        elif axis == "y":
            grper = axis_set[0].get_shared_y_axes()
        else:
            gen_util.accepted_values_error("axis", axis, ["x", "y"])

        # this did not work as a work-around to using get_shared_x_axes()
        # grper = mpl.cbook.Grouper(init=axis_set)

        grper.join(*axis_set)

    if adjust_tick_labels:
        adjust_tick_labels_for_sharing(axis_set, axes)

