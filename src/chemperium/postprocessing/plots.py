import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from chemperium.inp import InputArguments
from chemperium.gaussian.utils import geometry_type, gauss
from chemperium.postprocessing.metrics import PhysicalProperty
import numpy as np


def histogram_plot(values, feature, inp: InputArguments, num_bins: int = 200, alpha: float = 1.0):
    plt.figure()
    hfont = {"fontname": inp.font}
    plt.rc('font', size=inp.font_size)
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    plt.hist(values, num_bins, facecolor=inp.color_1, alpha=alpha)
    ax.set_ylabel("Occurrence", **hfont)
    for tick in ax.get_xticklabels():
        tick.set_fontname(inp.font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(inp.font)
    metric, unit = geometry_type(feature)
    ax.set_xlabel(str(metric + " [" + unit + "]"), **hfont)
    ax.set_title(feature, **hfont)
    save_location = inp.save_dir + "hist/" + feature + ".png"
    plt.savefig(save_location, bbox_inches="tight")


def gmm_plot(gmm_values, histogram_values, feature, inp: InputArguments):
    plt.figure()
    hfont = {"fontname": inp.font}
    plt.rc('font', size=inp.font_size)
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_ylabel("Occurrence", **hfont)
    for tick in ax.get_xticklabels():
        tick.set_fontname(inp.font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(inp.font)
    metric, unit = geometry_type(feature)
    ax.set_xlabel(str(metric + " [" + unit + "]"), **hfont)
    ax.set_title(feature, **hfont)
    value_range = np.arange(min(histogram_values), max(histogram_values), 0.001)

    for i in range(gmm_values.n_components):
        mu = gmm_values.means_[i][0]  # Mean of the i-th Gaussian
        sigma = np.sqrt(gmm_values.covariances_[i][0])
        gauss_curve = gauss(value_range, mu, sigma)
        plt.plot(value_range, gauss_curve, c=inp.color_1)
        plt.plot(mu, 1/(sigma*np.sqrt(2*np.pi)), "x", c=inp.color_2)

    save_location = inp.save_dir + "gmm/" + feature + ".png"
    plt.savefig(save_location, bbox_inches="tight")


def output_plot(rdkit_molecules: list, outputs: list):
    inp = InputArguments()
    prop = PhysicalProperty(inp.property)
    name_prop = prop.name
    metric_prop = prop.metric
    ha = np.array([molecule.GetNumHeavyAtoms() for molecule in rdkit_molecules])
    outputs = np.asarray(outputs).astype(np.float32)

    plt.figure()
    hfont = {"fontname": inp.font}
    plt.rc('font', size=inp.font_size)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_xlabel("Number of Heavy Atoms", **hfont)
    ax.set_ylabel("{} [{}]".format(name_prop, metric_prop))
    for tick in ax.get_xticklabels():
        tick.set_fontname(inp.font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(inp.font)
    ax.scatter(ha, outputs, s=20, c=inp.color_1, alpha=0.1)
    save_location = inp.save_dir + "output_plot.png"
    plt.savefig(save_location, bbox_inches="tight")


def performance_plot(true_values: np.ndarray, predictions: np.ndarray):
    inp = InputArguments()
    prop = PhysicalProperty(inp.property)
    name_prop = prop.name
    metric_prop = prop.metric

    plt.figure()
    hfont = {"fontname": inp.font}
    font = FontProperties(family=inp.font,
                          weight='normal',
                          style='normal', size=inp.font_size)
    plt.rc('font', size=inp.font_size)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_xlabel("True {} [{}]".format(name_prop, metric_prop), **hfont)
    ax.set_ylabel("Predicted {} [{}]".format(name_prop, metric_prop), **hfont)
    for tick in ax.get_xticklabels():
        tick.set_fontname(inp.font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(inp.font)
    ax.scatter(true_values, predictions, s=20, c=inp.color_1, alpha=0.1, label="Data points")
    ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)],
            c=inp.color_3, linewidth=1.5, label="Parity")
    save_location = inp.save_dir + "parity_plot.png"
    plt.legend(loc="best", prop=font)
    plt.savefig(save_location, bbox_inches="tight")
