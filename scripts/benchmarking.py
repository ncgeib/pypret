""" This module implements testing procedures for retrieval algorithms.
"""
import path_helper
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import EngFormatter
from pypret import (FourierTransform, Pulse, random_gaussian, random_pulse,
                    PNPS, material, Retriever, lib)
from pypret.graphics import plot_complex


def benchmark_retrieval(pulse, scheme, algorithm, additive_noise=0.0,
                        repeat=10, maxiter=300, verbose=False,
                        initial_guess="random_gaussian", **kwargs):
    """ Benchmarks a pulse retrieval algorithm. Uses the parameters from our
    paper.

    If you want to benchmark other pulses/configurations you can use the
    procedure below as a starting point.
    """
    # instantiate the result object
    res = SimpleNamespace()
    res.pulse = pulse.copy()
    res.original_spectrum = pulse.spectrum

    # split the scheme
    process, method = scheme.lower().split("-")

    if method == "miips":
        # MIIPS
        parameter = np.linspace(0.0, 2.0*np.pi, 128)  # delta in rad
        pnps = PNPS(pulse, method, process, gamma=22.5e-15, alpha=1.5 * np.pi)
    elif method == "dscan":
        # d-scan
        parameter = np.linspace(-0.025, 0.025, 128)  # insertion in m
        pnps = PNPS(pulse, method, process, material=material.BK7)
    elif method == "ifrog":
        # ifrog
        if process == "sd":
            parameter = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * 4)
        else:
            parameter = pulse.t  # delay in s
        pnps = PNPS(pulse, method, process)
    elif method == "frog":
        # frog
        parameter = pulse.t  # delay in s
        pnps = PNPS(pulse, method, process)
    elif method == "tdp":
        # d-scan
        parameter = np.linspace(pulse.t[0], pulse.t[-1], 128)  # delay in s
        pnps = PNPS(pulse, method, process, center=790e-9, width=10.6e-9)
    else:
        raise ValueError("Method not supported!")
    pnps.calculate(pulse.spectrum, parameter)
    measurement = pnps.trace

    # add noise
    std = measurement.data.max() * additive_noise
    measurement.data += std * np.random.normal(size=measurement.data.shape)

    ret = Retriever(pnps, algorithm, verbose=verbose, logging=True,
                    maxiter=maxiter, **kwargs)

    res.retrievals = []
    for i in range(repeat):
        if initial_guess == "random_gaussian":
            # create random Gaussian pulse
            random_gaussian(pulse, 50e-15, 0.3 * np.pi)
        elif initial_guess == "random":
            pulse.spectrum = (np.random.uniform(size=pulse.N) *
                              np.exp(2.0j * np.pi *
                                     np.random.uniform(size=pulse.N)))
        elif initial_guess == "original":
            pulse.spectrum = res.original_spectrum
        else:
            raise ValueError("Initial guess mode '%s' not supported." % initial_guess)
        ret.retrieve(measurement, pulse.spectrum)
        res.retrievals.append(ret.result(res.original_spectrum))

    return res


class RetrievalResultPlot:

    def __init__(self, retrieval_result, plot=True, **kwargs):
        rr = self.retrieval_result = retrieval_result
        if rr.pulse_original is None:
            raise ValueError("This plot requires an original pulse to compare"
                             " to.")
        if plot:
            self.plot(**kwargs)

    def plot(self, xaxis='wavelength', yaxis='intensity', limit=True,
             phase_blanking=False, phase_blanking_threshold=1e-3, show=True):
        rr = self.retrieval_result
        # reconstruct a pulse from that
        pulse = Pulse(rr.pnps.ft, rr.pnps.w0, unit="om")

        # construct the figure
        fig = plt.figure(figsize=(30.0/2.54, 20.0/2.54))
        gs1 = gridspec.GridSpec(2, 2)
        gs2 = gridspec.GridSpec(2, 6)
        ax1 = plt.subplot(gs1[0, 0])
        ax2 = plt.subplot(gs1[0, 1])
        ax3 = plt.subplot(gs2[1, :2])
        ax4 = plt.subplot(gs2[1, 2:4])
        ax5 = plt.subplot(gs2[1, 4:])
        ax12 = ax1.twinx()
        ax22 = ax2.twinx()

        # Plot in time domain
        pulse.spectrum = rr.pulse_original  # the test pulse
        li011, li012, samp, spha = plot_complex(pulse.t, pulse.field, ax1, ax12, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold,
                          amplitude_line="k-", phase_line="k--")
        pulse.spectrum = rr.pulse_retrieved  # the retrieved pulse
        li11, li12, samp, spha = plot_complex(pulse.t, pulse.field, ax1, ax12, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        li11.set_linewidth(3.0)
        li11.set_color("#1f77b4")
        li11.set_alpha(0.6)
        li12.set_linewidth(3.0)
        li12.set_color("#ff7f0e")
        li12.set_alpha(0.6)

        fx = EngFormatter(unit="s")
        ax1.xaxis.set_major_formatter(fx)
        ax1.set_title("time domain")
        ax1.set_xlabel("time")
        ax1.set_ylabel(yaxis)
        ax12.set_ylabel("phase (rad)")
        ax1.legend([li011, li11, li12], ["original", "intensity",
                   "phase"])

        # frequency domain
        if xaxis == "wavelength":
            x = pulse.wl
            unit = "m"
            label = "wavelength"
        elif xaxis == "frequency":
            x = pulse.w
            unit = " rad Hz"
            label = "frequency"
        # Plot in spectral domain
        li021, li022, samp, spha = plot_complex(x, rr.pulse_original, ax2, ax22, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold,
                          amplitude_line="k-", phase_line="k--")
        li21, li22, samp, spha = plot_complex(x, rr.pulse_retrieved, ax2, ax22, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        li21.set_linewidth(3.0)
        li21.set_color("#1f77b4")
        li21.set_alpha(0.6)
        li22.set_linewidth(3.0)
        li22.set_color("#ff7f0e")
        li22.set_alpha(0.6)

        fx = EngFormatter(unit=unit)
        ax2.xaxis.set_major_formatter(fx)
        ax2.set_title("frequency domain")
        ax2.set_xlabel(label)
        ax2.set_ylabel(yaxis)
        ax22.set_ylabel("phase (rad)")
        ax2.legend([li021, li21, li22], ["original", "intensity",
                   "phase"])

        axes = [ax3, ax4, ax5]
        sc = 1.0 / rr.trace_input.max()
        traces = [rr.trace_input * sc, rr.trace_retrieved * sc,
                  (rr.trace_input - rr.trace_retrieved) * sc]
        titles = ["input", "retrieved", "difference"]
        cmaps = ["nipy_spectral", "nipy_spectral", "RdBu"]
        md = rr.measurement
        for ax, trace, title, cmap in zip(axes, traces, titles, cmaps):
            x, y = lib.edges(rr.pnps.process_w), lib.edges(rr.parameter)
            im = ax.pcolormesh(x, y, trace, cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(md.labels[1])
            ax.set_ylabel(md.labels[0])
            fx = EngFormatter(unit=md.units[1])
            ax.xaxis.set_major_formatter(fx)
            fy = EngFormatter(unit=md.units[0])
            ax.yaxis.set_major_formatter(fy)
            ax.set_title(title)

        self.fig = fig
        self.ax1, self.ax2 = ax1, ax2
        self.ax12, self.ax22 = ax12, ax22
        self.li11, self.li12, self.li21, self.li22 = li11, li12, li21, li22
        self.ax3, self.ax4, self.ax5 = ax3, ax4, ax5

        if show:
            #gs.tight_layout(fig)
            gs1.update(left=0.05, right=0.95, top=0.9, bottom=0.1,
                      hspace=0.25, wspace=0.5)
            gs2.update(left=0.1, right=0.95, top=0.9, bottom=0.1,
                      hspace=0.5, wspace=1.0)
            plt.show()
