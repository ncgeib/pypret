""" This module implements several helper routines for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from . import lib


def plot_meshdata(ax, md, cmap="nipy_spectral"):
    x, y = lib.edges(md.axes[1]), lib.edges(md.axes[0])
    im = ax.pcolormesh(x, y, md.data, cmap=cmap)
    ax.set_xlabel(md.labels[0])
    ax.set_ylabel(md.labels[1])

    fx = EngFormatter(unit=md.units[0])
    ax.xaxis.set_major_formatter(fx)
    fy = EngFormatter(unit=md.units[1])
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
    if limit:
        ax.set_xlim(lib.limit(x, amp))

    if phase_blanking:
        amp, phase = lib.mask_phase(x, amp, phase, phase_blanking_threshold)

    li1, = ax.plot(x, amp, amplitude_line)
    li2, = ax2.plot(x, phase, phase_line)

    return li1, li2


class PulsePlot:

    def __init__(self, pulse, plot=True, **kwargs):
        self.pulse = pulse
        if plot:
            self.plot(**kwargs)

    def plot(self, xaxis='wavelength', yaxis='intensity', limit=True,
             phase_blanking=False, phase_blanking_threshold=1e-3, show=True):
        pulse = self.pulse

        fig, axs = plt.subplots(1, 2)
        ax1, ax2 = axs.flat
        ax12 = ax1.twinx()
        ax22 = ax2.twinx()

        # time domain
        li11, li12 = plot_complex(pulse.t, pulse.field, ax1, ax12, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        fx = EngFormatter(unit="s")
        ax1.xaxis.set_major_formatter(fx)
        ax1.set_title("time domain")
        ax1.set_xlabel("time")
        ax1.set_ylabel(yaxis)
        ax12.set_ylabel("phase (rad)")
        # frequency domain
        if xaxis == "wavelength":
            x = pulse.wl
            unit = "m"
            label = "wavelength"
        elif xaxis == "frequency":
            x = pulse.w
            unit = " rad Hz"
            label = "frequency"
        li21, li22 = plot_complex(x, pulse.spectrum, ax2, ax22, yaxis=yaxis,
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

        if show:
            fig.tight_layout()
            plt.show()
