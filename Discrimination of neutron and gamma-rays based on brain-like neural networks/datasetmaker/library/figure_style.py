import matplotlib.pyplot as plt 

def set_size(width, fraction=1, ratio=(5**.5 - 1) / 2):
   """Set figure dimensions to avoid scaling in LaTeX.

   Parameters
   ----------
   width: float
           Document textwidth or columnwidth in pts
   fraction: float, optional
           Fraction of the width which you wish the figure to            occupy

   Returns
   -------
   fig_dim: tuple
           Dimensions of figure in inches
   """
   # Width of figure (in pts)
   fig_width_pt = width * fraction

   # Convert from pt to inches
   inches_per_pt = 1 / 72.27

   # Golden ratio to set aesthetic figure height
   # https://disq.us/p/2940ij3
   #    golden_ratio = (5**.5 - 1) / 2

   # Figure width in inches
   fig_width_in = fig_width_pt * inches_per_pt
   # Figure height in inches
   fig_height_in = fig_width_in * ratio

   fig_dim = (fig_width_in, fig_height_in)
   print (fig_dim)
   return fig_dim 

def set_style(textwidth=345, columnwidth=0, linewidth=0, paperwidth=0, fraction=1, ratio=(5**.5 - 1) / 2):

    tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    'axes.titlesize': 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.autolimit_mode": "round_numbers",
    "figure.subplot.bottom": 0.17,
    "figure.subplot.right": 0.95,
    "figure.subplot.left": 0.15,
    "figure.figsize": set_size(textwidth, fraction, ratio) # or above alternatives
    } 
    plt.rcParams.update(tex_fonts)