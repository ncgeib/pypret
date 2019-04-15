""" This module implements several helper routines for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from . import lib
from .frequencies import convert


def plot_meshdata(ax, md, cmap="nipy_spectral"):
    x, y = lib.edges(md.axes[1]), lib.edges(md.axes[0])
    im = ax.pcolormesh(x, y, md.data, cmap=cmap)
    ax.set_xlabel(md.labels[1])
    ax.set_ylabel(md.labels[0])

    fx = EngFormatter(unit=md.units[1])
    ax.xaxis.set_major_formatter(fx)
    fy = EngFormatter(unit=md.units[0])
    ax.yaxis.set_major_formatter(fy)
    return im


class MeshDataPlot:

    def __init__(self, mesh_data, plot=True, **kwargs):
        self.md = mesh_data
        if plot:
            self.plot(**kwargs)

    def plot(self, show=True):
        md = self.md

        fig, ax = plt.subplots()
        im = plot_meshdata(ax, md, "nipy_spectral")
        fig.colorbar(im, ax=ax)

        self.fig, self.ax = fig, ax
        self.im = im
        if show:
            fig.tight_layout()
            plt.show()

    def show(self):
        plt.show()


def plot_complex(x, y, ax, ax2, yaxis='intensity', limit=False,
                 phase_blanking=False, phase_blanking_threshold=1e-3,
                 amplitude_line="r-", phase_line="b-"):
    if yaxis == "intensity":
        amp = lib.abs2(y)
    elif yaxis == "amplitude":
        amp = np.abs(y)
    else:
        raise ValueError("yaxis mode '%s' is unknown!" % yaxis)
    phase = lib.phase(y)
    # center phase by weighted mean
    phase -= lib.mean(phase, amp * amp)
    if phase_blanking:
        x2, phase2 = lib.mask_phase(x, amp, phase, phase_blanking_threshold)
    else:
        x2, phase2 = x, phase
    if limit:
        xlim = lib.limit(x, amp)
        ax.set_xlim(xlim)
        f = (x2 >= xlim[0]) & (x2 <= xlim[1])
        ax2.set_ylim(lib.limit(phase2[f], padding=0.05))

    li1, = ax.plot(x, amp, amplitude_line)
    li2, = ax2.plot(x2, phase2, phase_line)

    return li1, li2, amp, phase


class PulsePlot:

    def __init__(self, pulse, plot=True, **kwargs):
        self.pulse = pulse
        if plot:
            self.plot(**kwargs)

    def plot(self, xaxis='wavelength', yaxis='intensity', limit=True,
             oversampling=False, phase_blanking=False,
             phase_blanking_threshold=1e-3, show=True):
        pulse = self.pulse

        fig, axs = plt.subplots(1, 2)
        ax1, ax2 = axs.flat
        ax12 = ax1.twinx()
        ax22 = ax2.twinx()

        if oversampling:
            t = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * oversampling)
            field = pulse.field_at(t)
        else:
            t = pulse.t
            field = pulse.field

        # time domain
        li11, li12, tamp, tpha = plot_complex(t, field, ax1, ax12, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        fx = EngFormatter(unit="s")
        ax1.xaxis.set_major_formatter(fx)
        ax1.set_title("time domain")
        ax1.set_xlabel("time")
        ax1.set_ylabel(yaxis)
        ax12.set_ylabel("phase (rad)")

        # frequency domain
        if oversampling:
            w = np.linspace(pulse.w[0], pulse.w[-1], pulse.N * oversampling)
            spectrum = pulse.spectrum_at(w)
        else:
            w = pulse.w
            spectrum = pulse.spectrum

        if xaxis == "wavelength":
            w = convert(w + pulse.w0, "om", "wl")
            unit = "m"
            label = "wavelength"
        elif xaxis == "frequency":
            w = w
            unit = " rad Hz"
            label = "frequency"

        li21, li22, samp, spha = plot_complex(w, spectrum, ax2, ax22, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        fx = EngFormatter(unit=unit)
        ax2.xaxis.set_major_formatter(fx)
        ax2.set_title("frequency domain")
        ax2.set_xlabel(label)
        ax2.set_ylabel(yaxis)
        ax22.set_ylabel("phase (rad)")

        self.fig = fig
        self.ax1, self.ax2 = ax1, ax2
        self.ax12, self.ax22 = ax12, ax22
        self.li11, self.li12, self.li21, self.li22 = li11, li12, li21, li22
        self.tamp, self.tpha = tamp, tpha
        self.samp, self.spha = samp, spha

        if show:
            fig.tight_layout()
            plt.show()
