import logging
import warnings
from abc import ABC
from dataclasses import dataclass

import astropy.constants as c
import astropy.units as u
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.visualization import ImageNormalize, ZScaleInterval
from scipy.signal import correlate

from dstools.rm import PolObservation

logger = logging.getLogger(__name__)

COLORS = {
    "I": "firebrick",
    "Q": "lightgreen",
    "U": "cornflowerblue",
    "V": "darkorange",
}


def slice_array(a, ax1_min, ax1_max, ax2_min=None, ax2_max=None):
    """Slice 1D or 2D array with variable lower and upper boundaries."""

    if ax2_min and ax2_max:
        a = a[ax1_min:, :] if ax1_max == 0 else a[ax1_min:ax1_max, :]
        a = a[:, ax2_min:] if ax2_max == 0 else a[:, ax2_min:ax2_max]
    else:
        a = a[ax1_min:] if ax1_max == 0 else a[ax1_min:ax1_max]

    return a


@dataclass
class DynamicSpectrum:
    project: str
    prefix: str = ""
    band: str = "AT_L"

    favg: int = 1
    tavg: int = 1

    minfreq: float = None
    maxfreq: float = None
    mintime: float = None
    maxtime: float = None

    tunit: u.Quantity = u.hour
    scantime: float = 10.1

    fold: bool = False
    period: float = None
    period_offset: float = 0.0
    fold_periods: int = 2

    calscans: bool = True
    trim: bool = True

    def __post_init__(self):
        self.src = self.project.split("/")[-1]

        self.ds_path = f"{self.project}/dynamic_spectra/{self.band}"
        self._timelabel = "Phase" if self.fold else f"Time ({self.tunit})"

        self._load_data()
        (
            self.avg_scan_dt,
            scan_start_indices,
            scan_end_indices,
        ) = self._get_scan_intervals()

        self._stack_cal_scans(scan_start_indices, scan_end_indices)

        self._make_stokes()

        self.tmin = self.time[0]
        self.tmax = self.time[-1]
        self.fmin = self.freq[0]
        self.fmax = self.freq[-1]

        tbins = self.data["I"].shape[0]
        fbins = self.data["I"].shape[1]
        self.time_resolution = (self.tmax - self.tmin) * self.tunit / tbins
        self.freq_resolution = (self.fmax - self.fmin) * u.MHz / fbins

    def _fold(self, data):
        """Average chunks of data folding at specified period."""

        # Calculate number of pixels in each chunk
        pixel_duration = self.avg_scan_dt * self.tavg
        chunk_length = min(int(self.period // pixel_duration), len(data))

        # Create left-padded nans, derived from period phase offset
        offset = (0.5 + self.period_offset) * self.period
        leftpad_length = int(offset // pixel_duration)
        leftpad_chunk = np.full((leftpad_length, data.shape[1]), np.nan)

        # Create right-padded nans
        rightpad_length = chunk_length - (leftpad_length + len(data)) % chunk_length
        rightpad_chunk = np.full((rightpad_length, data.shape[1]), np.nan)

        # Stack and split data
        data = np.vstack((leftpad_chunk, data, rightpad_chunk))
        numsplits = int(data.shape[0] // chunk_length)
        arrays = np.split(data, numsplits)

        # Compute average along stack axis
        data = np.nanmean(arrays, axis=0)

        return np.tile(data, (self.fold_periods, 1))

    def _get_scan_intervals(self):
        """Find indices of start/end of each calibrator scan cycle."""

        dts = [0]
        dts.extend([self.time[i] - self.time[i - 1] for i in range(1, len(self.time))])
        dts = np.array(dts)

        # Locate indices signaling beginning of cal-scan (scan intervals longer than ~10 seconds)
        scan_start_indices = np.where(np.abs(dts) > self.scantime)[0]

        # End indices are just prior to the next start index, then
        scan_end_indices = scan_start_indices - 1

        # Insert first scan start index and last scan end index
        scan_start_indices = np.insert(scan_start_indices, 0, 0)
        scan_end_indices = np.append(scan_end_indices, len(self.time) - 1)

        # Calculate average correlator cycle time from first scan
        avg_scan_dt = np.median(dts[: scan_end_indices[0]])

        return avg_scan_dt, scan_start_indices, scan_end_indices

    def _load_data(self):
        """Load instrumental pols and time/freq data, converting to MHz, s, and mJy."""

        # Import instrumental polarisations and time/frequency arrays
        file_prefix = f"{self.ds_path}/{self.prefix}"

        XX = np.load(f"{file_prefix}dynamic_spectra_XX.npy", allow_pickle=True) * 1e3
        XY = np.load(f"{file_prefix}dynamic_spectra_XY.npy", allow_pickle=True) * 1e3
        YX = np.load(f"{file_prefix}dynamic_spectra_YX.npy", allow_pickle=True) * 1e3
        YY = np.load(f"{file_prefix}dynamic_spectra_YY.npy", allow_pickle=True) * 1e3

        freq = np.load(f"{file_prefix}freq.npy", allow_pickle=True) / 1e6
        time = np.load(f"{file_prefix}time.npy", allow_pickle=True)

        # Set timescale
        time_scale_factor = self.tunit.to(u.s)
        time /= time_scale_factor
        self.scantime /= time_scale_factor

        # Flip ATCA L-band frequency axis to intuitive order
        if self.band == "AT_L":
            XX = np.flip(XX, axis=1)
            XY = np.flip(XY, axis=1)
            YX = np.flip(YX, axis=1)
            YY = np.flip(YY, axis=1)

            freq = np.flip(freq)

        # Optionally remove flagged channels at top/bottom of band
        if self.trim:
            full = np.nansum((XX + XY + YX + YY), axis=0)
            full[full == 0.0 + 0.0j] = np.nan

            allpols = np.isfinite(full)
            minchan = np.argmax(allpols)
            maxchan = -np.argmax(allpols[::-1]) + 1
        else:
            minchan = 0
            maxchan = 0

        # Select channel range
        if self.minfreq:
            minchan = -np.argmax((freq < self.minfreq)[::-1]) - 1
        if self.maxfreq:
            maxchan = np.argmax(freq > self.maxfreq)

        # Select time range
        if self.mintime:
            mintime = np.argmax(time - time[0] > self.mintime)
        else:
            mintime = 0

        if self.maxtime:
            maxtime = -np.argmax((time - time[0] < self.maxtime)[::-1]) + 1
        else:
            maxtime = 0

        # Make data selection
        self.XX = slice_array(XX, mintime, maxtime, minchan, maxchan)
        self.XY = slice_array(XY, mintime, maxtime, minchan, maxchan)
        self.YX = slice_array(YX, mintime, maxtime, minchan, maxchan)
        self.YY = slice_array(YY, mintime, maxtime, minchan, maxchan)

        self.freq = slice_array(freq, minchan, maxchan)
        self.time = slice_array(time, mintime, maxtime)

        # Identify start time and set observation start to t=0
        self.time_start = Time(
            self.time[0] * time_scale_factor / 3600 / 24,
            format="mjd",
            scale="utc",
        )
        self.time_start.format = "iso"

        self.time -= time[0]

    def _stack_cal_scans(self, scan_start_indices, scan_end_indices):
        """Insert null data representing off-source time."""

        # Calculate number of cycles in each calibrator/stow break
        time_end_break = self.time[scan_start_indices[1:]]
        time_start_break = self.time[scan_end_indices[:-1]]
        num_break_cycles = (
            np.append((time_end_break - time_start_break), 0) / self.avg_scan_dt
        )
        num_channels = self.XX.shape[1]

        # Create initial time-slice to start stacking target and calibrator scans together
        new_data_XX = new_data_XY = new_data_YX = new_data_YY = np.zeros(
            (1, num_channels),
            dtype=complex,
        )

        # Check that data is not being truncated by stacking insufficient time chunks together
        # This is a diagnostic of errors in data combination, where duplicated times are
        # dropped in dstools.make_dspec if data is combined incorrectly.
        if scan_end_indices[-1] + 1 < self.XX.shape[0]:
            logger.warning("More time samples in data array than time array")
            logger.warning("Check results with -C to see full data")

        for start_index, end_index, num_scans in zip(
            scan_start_indices, scan_end_indices, num_break_cycles
        ):
            # Select each contiguous on-target chunk of data
            XX_chunk = self.XX[start_index : end_index + 1, :]
            XY_chunk = self.XY[start_index : end_index + 1, :]
            YX_chunk = self.YX[start_index : end_index + 1, :]
            YY_chunk = self.YY[start_index : end_index + 1, :]

            # Make array of complex NaN's for subsequent calibrator / stow gaps
            # and append to each on-target chunk of data
            if self.calscans:
                num_nans = (int(round(num_scans)), num_channels)
                nan_chunk = np.full(num_nans, np.nan + np.nan * 1j)

                XX_chunk = np.ma.vstack([XX_chunk, nan_chunk])
                XY_chunk = np.ma.vstack([XY_chunk, nan_chunk])
                YX_chunk = np.ma.vstack([YX_chunk, nan_chunk])
                YY_chunk = np.ma.vstack([YY_chunk, nan_chunk])

            new_data_XX = np.ma.vstack([new_data_XX, XX_chunk])
            new_data_XY = np.ma.vstack([new_data_XY, XY_chunk])
            new_data_YX = np.ma.vstack([new_data_YX, YX_chunk])
            new_data_YY = np.ma.vstack([new_data_YY, YY_chunk])

        new_data_XX = new_data_XX[1:]
        new_data_XY = new_data_XY[1:]
        new_data_YX = new_data_YX[1:]
        new_data_YY = new_data_YY[1:]

        tbins = len(new_data_XX) // self.tavg
        fbins = num_channels // self.favg

        self.XX = self.rebin2D(new_data_XX, (tbins, fbins))
        self.YX = self.rebin2D(new_data_YX, (tbins, fbins))
        self.XY = self.rebin2D(new_data_XY, (tbins, fbins))
        self.YY = self.rebin2D(new_data_YY, (tbins, fbins))

    def _make_stokes(self):
        """Convert instrumental polarisations to Stokes products."""

        # Compute Stokes products from instrumental pols
        I = (self.XX + self.YY) / 2
        Q = (self.XX - self.YY) / 2
        U = (self.XY + self.YX) / 2
        V = 1j * (self.YX - self.XY) / 2

        # Fold data to selected period
        if self.fold:
            if not self.period:
                raise ValueError("Must pass period argument when folding.")

            I = self._fold(I)
            Q = self._fold(Q)
            U = self._fold(U)
            V = self._fold(V)

        self.data = {"I": I, "Q": Q, "U": U, "V": V}

    def acf(self, stokes):
        """Generate a 2D auto-correlation of the dynamic spectrum."""

        # Replace NaN with zeros to calculate auto-correlation
        data = self.data[stokes].real.copy()
        data[np.isnan(data)] = 0.0

        # Compute auto-correlation and select upper-right quadrant
        acf2d = correlate(data, data)
        acf2d = acf2d[acf2d.shape[0] // 2 :, acf2d.shape[1] // 2 :]

        # Reorder time-frequency axes and normalise
        acf2d = np.flip(acf2d, axis=1).T
        acf2d /= np.nanmax(acf2d)

        return acf2d

    def rebin(self, o, n, axis):
        """Create unitary array compression matrix from o -> n length.

        if rebinning along row axis we want:
         - (o // n) + 1 entries in each row that sum to unity,
         - each column to sum to the compression ratio o / n
         - values distributed along the row in units of o / n until expired

           >>> rebin(5, 3)
           array([[0.6, 0.4, 0. , 0. , 0. ],
                  [0. , 0.2, 0.6, 0.2, 0. ],
                  [0. , 0. , 0. , 0.4, 0.6]])

         - transpose of this for column rebinning

        The inner product of this compressor with an array will rebin
        the array conserving the total intensity along the given axis.
        """

        compressor = np.zeros((n, o))

        # Exit early with empty array if chunk is empty
        if compressor.size == 0:
            return compressor

        comp_ratio = n / o

        nrow = 0
        ncol = 0

        budget = 1
        overflow = 0

        # While loop to avoid visiting n^2 zero-value cells
        while nrow < n and ncol < o:
            # Use overflow if just spilled over from last row
            if overflow > 0:
                value = overflow
                overflow = 0
                budget -= value
                row_shift = 0
                col_shift = 1
            # Use remaining budget if at end of current row
            elif budget < comp_ratio:
                value = budget
                overflow = comp_ratio - budget
                budget = 1
                row_shift = 1
                col_shift = 0
            # Otherwise spend n / o and move to next column
            else:
                value = comp_ratio
                budget -= value
                row_shift = 0
                col_shift = 1

            compressor[nrow, ncol] = value
            nrow += row_shift
            ncol += col_shift

        return compressor if axis == 0 else compressor.T

    def rebin2D(self, array, new_shape):
        """Re-bin along time / frequency axes conserving flux."""

        if new_shape == array.shape:
            array[array == 0 + 0j] = np.nan
            return array

        if new_shape[0] > array.shape[0] or new_shape[1] > array.shape[1]:
            raise ValueError(
                "New shape should not be greater than old shape in either dimension"
            )

        time_comp = self.rebin(array.shape[0], new_shape[0], axis=0)
        freq_comp = self.rebin(array.shape[1], new_shape[1], axis=1)
        array[np.isnan(array) | array.mask] = 0 + 0j
        result = time_comp @ np.array(array) @ freq_comp
        result[result == 0 + 0j] = np.nan

        return result

    def plot_rmsynth(self):
        I = self.data["I"]
        Q = self.data["Q"]
        U = self.data["U"]

        angle = 0.5 * np.arctan2(U.real, Q.real)

        # Mask based on Stokes I RMS
        mask = np.abs(I.real) < 1 * np.nanstd(I.imag)
        angle[mask] = np.nan

        # RM synthesis
        I[np.isnan(I)] = 0 + 0j
        Q[np.isnan(Q)] = 0 + 0j
        U[np.isnan(U)] = 0 + 0j

        tslice = np.argmax(np.nanmean(I.real, axis=1))
        stokes_arrays = (I[tslice, :].real, Q[tslice, :].real, U[tslice, :].real)
        phi_axis = np.arange(-2000.0, 2000.1, 0.1)
        p = PolObservation(
            self.freq * 1e6,
            stokes_arrays,
            verbose=False,
        )
        p.rmsynthesis(phi_axis, verbose=False)
        p.rmclean(
            cutoff=1.0,
            verbose=False,
        )

        RM = p.phi[np.argmax(abs(p.fdf))]
        logger.info(f"Peak RM of {RM:.1f} rad/m2")

        # Correct for Faraday Rotation
        for chan in range(angle.shape[1]):
            lam = (c.c / ((self.fmin + chan) * u.MHz)).to(u.m).value
            angle[:, chan] -= RM * lam**2

            # Clamp to -pi/2 to pi/2
            angle = np.arctan(np.tan(angle))

        bins = self.data["I"].shape[0]

        # Frequency-averaged polarisation angle lightcurve
        chi_lc_fig, chi_lc_ax = plt.subplots(figsize=(7, 5))

        # Calculate lambda
        interval = (self.tmax - self.tmin) / bins
        time = np.array([self.tmin + i * interval for i in range(bins)])
        angle_lc = np.nanmean(angle, axis=1)

        # RM Synthesis FDF
        fdf_fig, fdf_ax = plt.subplots(figsize=(7, 5))
        fdf_ax.plot(p.rmsf_phi, np.abs(p.rmsf), color="k", label="RMSF")
        fdf_ax.plot(p.phi, np.abs(p.fdf), color="r", label="FDF")
        fdf_ax.plot(p.phi, np.abs(p.rm_cleaned), color="b", label="Clean")
        fdf_ax.plot(p.phi, np.abs(p.rm_comps), color="g", label="Model")
        fdf_ax.set_xlabel(r"RM ($rad/m^2$)")
        fdf_ax.set_ylabel("Amplitude")
        fdf_ax.legend()

        # Polarisation angle LC
        chi_lc_ax.scatter(
            time,
            angle_lc,
            color="k",
            s=2,
        )

        chi_lc_ax.set_xlabel("Time (min)")
        chi_lc_ax.set_ylabel("PA")

        # Polarisation angle DS
        chi_ds_fig, chi_ds_ax = plt.subplots(figsize=(7, 5))

        norm = ImageNormalize(angle, interval=ZScaleInterval(contrast=0.2))

        im = chi_ds_ax.imshow(
            angle.T,
            extent=[self.tmin, self.tmax, self.fmin, self.fmax],
            aspect="auto",
            origin="lower",
            norm=norm,
            cmap="coolwarm",
        )

        # Colorbar and labels
        chi_ds_ax.set_xlabel(f"Time ({self.tunit})")
        chi_ds_ax.set_ylabel("Frequency (MHz)")
        cb = chi_ds_fig.colorbar(im, ax=chi_ds_ax, fraction=0.05, pad=0.02)
        cb.set_label("PA")

        return chi_ds_ax, chi_ds_fig, chi_lc_fig, chi_lc_ax, fdf_ax, fdf_fig

    def plot_crosspol_ds(self, cmax=20, cmin=0):
        """Plot quadrature sum of cross-polarisations: sqrt(U^2 + V^2)."""

        U = self.data["U"]
        V = self.data["V"]
        data = np.sqrt(np.abs(U**2 + V**2))

        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.2))

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            np.transpose(data),
            extent=[self.tmin, self.tmax, self.fmin, self.fmax],
            aspect="auto",
            origin="lower",
            norm=norm,
            clim=(cmin, cmax),
            cmap="plasma",
        )
        ax.text(
            0.05,
            0.95,
            r"sqrt(U^2 + V^2)",
            color="white",
            weight="heavy",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            transform=ax.transAxes,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("Flux Density (mJy)")
        ax.set_xlabel(self._timelabel)
        ax.set_ylabel("Frequency (MHz)")

        return fig, ax

    def plot_lightcurve(self, stokes):
        """Plot channel-averaged lightcurve."""

        lc = LightCurve(self, stokes)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig, ax = lc.plot(fig, ax)

        return fig, ax

    def plot_spectrum(self, stokes):
        """Plot time-averaged spectrum."""

        sp = Spectrum(self, stokes)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig, ax = sp.plot(fig, ax)

        return fig, ax

    def plot_ds(self, stokes, cmax=20, imag=False):
        """Plot dynamic spectrum."""

        data = self.data[stokes].imag if imag else self.data[stokes].real

        phasemax = 0.5 * self.fold_periods
        tmin, tmax = (-phasemax, phasemax) if self.fold else (self.tmin, self.tmax)
        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.2))
        cmap = "plasma" if stokes == "I" else "coolwarm"
        cmin = -2 if stokes == "I" else -cmax
        cmax = cmax

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            np.transpose(data),
            extent=[tmin, tmax, self.fmin, self.fmax],
            aspect="auto",
            origin="lower",
            norm=norm,
            clim=(cmin, cmax),
            cmap=cmap,
        )
        ax.text(
            0.05,
            0.95,
            f"Stokes {stokes}",
            color="white",
            weight="heavy",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            transform=ax.transAxes,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("Flux Density (mJy)")

        ax.set_xlabel(self._timelabel)
        ax.set_ylabel("Frequency (MHz)")

        return fig, ax

    def plot_acf(self, stokes="I", contrast=0.4):
        """Plot 2D auto-correlation function of the dynamic spectrum."""

        acf2d = self.acf(stokes)

        # Plot 2D ACF
        acf_fig, acf_ax = plt.subplots(figsize=(7, 5))

        norm = ImageNormalize(acf2d, interval=ZScaleInterval(contrast=contrast))
        im = acf_ax.imshow(
            acf2d,
            extent=[0, self.tmax - self.tmin, 0, self.fmax - self.fmin],
            aspect="auto",
            norm=norm,
            cmap="plasma",
        )
        cb = acf_fig.colorbar(
            im,
            ax=acf_ax,
            fraction=0.05,
            pad=0.02,
        )
        cb.formatter.set_powerlimits((0, 0))
        cb.set_label("ACF")

        acf_ax.set_xlabel("Time Lag (h)")
        acf_ax.set_ylabel("Frequency Lag (MHz)")

        # Plot zero frequency lag trace
        acfz_fig, acfz_ax = plt.subplots(figsize=(7, 5))

        zero_trace_acf = acf2d[-1, 1:]
        time_lag = np.linspace(0, self.tmax - self.tmin, len(zero_trace_acf))
        acfz_ax.plot(
            time_lag,
            zero_trace_acf,
            color="k",
        )

        acfz_ax.set_xlabel("Time Lag (h)")
        acfz_ax.set_ylabel("ACF")

        return acf_fig, acf_ax, acfz_fig, acfz_ax


class TimeFreqSeries(ABC):
    def _construct_yaxis(self, avg_axis):
        """Construct y-axis averaging flux over x-axis."""

        # Catch RuntimeWarning that occurs when averaging empty time/freq slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y = {
                stokes: np.nanmean(self.ds.data[stokes], axis=avg_axis)
                for stokes in self.stokes
            }
            sqrtn = np.sqrt(self.ds.data["I"].shape[1])
            yerr = {
                stokes: np.nanstd(self.ds.data[stokes].imag, axis=avg_axis) / sqrtn
                for stokes in self.stokes
            }

        return y, yerr

    def plot(self, avg_axis):
        # Overplot each specified polarisation
        for stokes in self.stokes:
            # Plot time/frequency series
            self.ax.errorbar(
                self.x,
                self.y[stokes].real,
                yerr=self.yerr[stokes],
                lw=1,
                color=COLORS[stokes],
                marker="o",
                markersize=1,
                label=stokes,
            )

        self.ax.set_ylabel("Flux Density (mJy)")
        self.ax.legend()

        return

    def save(self, savepath):
        values = self.valstart + self.x * self.unit

        df = pd.DataFrame({self.column: values})
        for stokes in self.stokes:
            df[f"flux_density_{stokes}"] = self.y[stokes].real.reshape(1, -1)[0]
            df[f"flux_density_{stokes}_err"] = self.yerr[stokes]

        df.dropna().to_csv(savepath, index=False)

        return


@dataclass
class LightCurve(TimeFreqSeries):
    ds: DynamicSpectrum
    stokes: str

    def __post_init__(self):
        self.column = "time"
        self.unit = self.ds.tunit
        self.valstart = self.ds.time_start

        # Set time/phase axis limits if folded
        phasemax = 0.5 * self.ds.fold_periods
        valmin, valmax = (
            (-phasemax, phasemax) if self.ds.fold else (self.ds.tmin, self.ds.tmax)
        )

        # Construct time and flux axes
        bins = self.ds.data["I"].shape[0]
        interval = (valmax - valmin) / bins
        self.x = np.array([valmin + i * interval for i in range(bins)])
        self.y, self.yerr = self._construct_yaxis(avg_axis=1)

    def plot(self, fig, ax):
        self.fig = fig
        self.ax = ax

        # Set time axis label
        self.ax.set_xlabel(self.ds._timelabel)

        # Plot with lightcurve/spectrum independent parameters
        super().plot(avg_axis=1)

        return self.fig, self.ax


@dataclass
class Spectrum(TimeFreqSeries):
    ds: DynamicSpectrum
    stokes: str

    def __post_init__(self):
        self.column = "frequency"
        self.unit = u.MHz
        self.valstart = 0

        # Construct frequency axis
        bins = self.ds.data["I"].shape[1]
        interval = (self.ds.fmax - self.ds.fmin) / bins
        self.x = np.array([self.ds.fmin + i * interval for i in range(bins)])
        self.y, self.yerr = self._construct_yaxis(avg_axis=0)

    def plot(self, fig, ax):
        self.fig = fig
        self.ax = ax

        # Set frequency axis label
        ax.set_xlabel("Frequency (MHz)")

        # Plot with lightcurve/spectrum independent parameters
        super().plot(avg_axis=0)

        return self.fig, self.ax
