import logging
import warnings
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.visualization import ZScaleInterval, ImageNormalize
from dataclasses import dataclass
from scipy.signal import correlate

logger = logging.getLogger(__name__)

@dataclass
class DynamicSpectrum:

    project: str
    prefix: str=''
    band: str='L'

    favg: int=1
    tavg: int=1

    period: float=None
    period_offset: float=0.

    calscans: bool=True
    fold: bool=False
    trim: bool=True
    save: bool=False

    def __post_init__(self):
        self.src = self.project.split('/')[-1]

        self.ds_path = f'{self.project}/dynamic_spectra/{self.band}'

        self._load_data()
        self.avg_scan_dt, scan_start_indices, scan_end_indices = self._get_scan_intervals()

        if self.calscans:
            self._stack_cal_scans(scan_start_indices, scan_end_indices)

        self._make_stokes()

        self.tmin = self.time[0]
        self.tmax = self.time[-1]
        self.fmin = self.freq[-1]
        self.fmax = self.freq[0]

    def _fold(self, data):
        '''Average chunks of data folding at specified period.'''

        # Calculate number of pixels in each chunk
        pixel_duration = self.avg_scan_dt * self.tavg
        chunk_length = min(int(self.period // pixel_duration), len(data))

        # Create left-padded zeros, derived from period phase offset
        offset = (0.5 + self.period_offset) * self.period 
        leftpad_length = int(offset // self.avg_scan_dt)
        leftpad_chunk = np.zeros((leftpad_length, data.shape[1]))

        # Create right-padded zeros
        rightpad_length = chunk_length - (leftpad_length + len(data)) % chunk_length
        rightpad_chunk = np.zeros((rightpad_length, data.shape[1]))

        # Stack and split data
        data = np.vstack((leftpad_chunk, data, rightpad_chunk))
        numsplits = int(data.shape[0] // chunk_length)
        arrays = np.split(data, numsplits)

        # Compute average along stack axis
        data = np.nanmean(arrays, axis=0)

        return data

    def _get_scan_intervals(self):
        '''Find indices of start/end of each calibrator scan cycle.'''

        dts = [0]
        dts.extend([self.time[i] - self.time[i-1] for i in range(1, len(self.time))])
        dts = np.array(dts)

        # Locate indices signaling beginning of cal-scan (scan intervals longer than ~10 seconds)
        scan_start_indices = np.where(np.abs(dts) > 10.1 / 3600)[0]

        # End indices are just prior to the next start index, then 
        scan_end_indices = scan_start_indices - 1

        # Insert first scan start index and last scan end index
        scan_start_indices = np.insert(scan_start_indices, 0, 0)
        scan_end_indices = np.append(scan_end_indices, len(self.time) - 1)

        # Calculate average correlator cycle time from first scan
        avg_scan_dt = np.median(dts[: scan_end_indices[0]])

        return avg_scan_dt, scan_start_indices, scan_end_indices

    def _load_data(self):
        '''Load instrumental pols and time/freq data, converting to MHz, s, and mJy.'''

        file_prefix = f'{self.ds_path}/{self.prefix}'

        XX = np.load(f'{file_prefix}dynamic_spectra_XX.npy', allow_pickle=True) * 1e3
        XY = np.load(f'{file_prefix}dynamic_spectra_XY.npy', allow_pickle=True) * 1e3
        YX = np.load(f'{file_prefix}dynamic_spectra_YX.npy', allow_pickle=True) * 1e3
        YY = np.load(f'{file_prefix}dynamic_spectra_YY.npy', allow_pickle=True) * 1e3

        # Remove flagged channels at top/bottom of band
        if self.trim:
            allpols = np.isfinite(np.nansum((XX + XY + YX + YY), axis=0))
            minfreq = np.argmax(allpols)
            maxfreq = -np.argmax(allpols[::-1]) + 1
        else:
            minfreq = 0
            maxfreq = -1

        self.XX = XX[:, minfreq:maxfreq]
        self.XY = XY[:, minfreq:maxfreq]
        self.YX = YX[:, minfreq:maxfreq]
        self.YY = YY[:, minfreq:maxfreq]

        freq = np.load(f'{file_prefix}freq.npy', allow_pickle=True) / 1e6
        self.freq = freq[minfreq:maxfreq]
        self.time = np.load(f'{file_prefix}time.npy', allow_pickle=True) / 3600

        self.time_start = Time(self.time[0] / 24.0, format='mjd', scale='utc')
        self.time_start.format = 'iso'

        self.time -= self.time[0]
        

    def _stack_cal_scans(self, scan_start_indices, scan_end_indices):
        '''Insert null data representing off-source time.'''

        # Calculate number of cycles in each calibrator/stow break
        time_end_break = self.time[scan_start_indices[1:]]
        time_start_break = self.time[scan_end_indices[:-1]]
        num_break_cycles = np.append((time_end_break - time_start_break), 0) / self.avg_scan_dt
        num_channels = self.XX.shape[1]

        fbins = num_channels // self.favg

        # Create initial time-slice to start stacking target and calibrator scans together
        new_data_XX = new_data_XY = new_data_YX = new_data_YY = np.zeros(
            (1, fbins), dtype=complex
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

            tbins = (end_index - start_index) // self.tavg

            # Select each contiguous on-target chunk of data
            XX_chunk = self.XX[start_index:end_index+1, :]
            XY_chunk = self.XY[start_index:end_index+1, :]
            YX_chunk = self.YX[start_index:end_index+1, :]
            YY_chunk = self.YY[start_index:end_index+1, :]
            
            # Rebin data with selected time and frequency averaging factors
            XX_chunk = self.rebin2D(XX_chunk, (tbins, fbins))
            XY_chunk = self.rebin2D(XY_chunk, (tbins, fbins))
            YX_chunk = self.rebin2D(YX_chunk, (tbins, fbins))
            YY_chunk = self.rebin2D(YY_chunk, (tbins, fbins))

            # Make array of complex NaN's for subsequent calibrator / stow gaps
            # and append to each on-target chunk of data
            nan_chunk = np.full((int(num_scans) // self.tavg, fbins), np.nan + np.nan * 1j)

            new_data_XX = np.vstack([new_data_XX, np.vstack([XX_chunk, nan_chunk])])
            new_data_XY = np.vstack([new_data_XY, np.vstack([XY_chunk, nan_chunk])])
            new_data_YX = np.vstack([new_data_YX, np.vstack([YX_chunk, nan_chunk])])
            new_data_YY = np.vstack([new_data_YY, np.vstack([YY_chunk, nan_chunk])])

        # Mask out NaN values
        self.XX = new_data_XX[1:]
        self.XY = new_data_XY[1:]
        self.YX = new_data_YX[1:]
        self.YY = new_data_YY[1:]

    def _make_stokes(self):
        '''Convert instrumental polarisations to Stokes products.'''

        # Compute Stokes products from instrumental pols
        I = (self.XX + self.YY) / 2
        Q = (self.XX - self.YY) / 2
        U = (self.XY + self.YX) / 2
        V = 1j * (self.XY - self.YX) / 2

        # Flip frequency axis of data to intuitive order
        I = np.flip(I, axis=1)
        Q = np.flip(Q, axis=1)
        U = np.flip(U, axis=1)
        V = np.flip(V, axis=1)

        # Set masked values to NaN (chunk stacking causes these to be complex zeros)
        I[I == 0.0 + 0.0j] = np.nan
        Q[Q == 0.0 + 0.0j] = np.nan
        U[U == 0.0 + 0.0j] = np.nan
        V[V == 0.0 + 0.0j] = np.nan

        # Fold data to selected period
        if self.fold:

            if not self.period:
                raise ValueError("Must pass period argument when folding.")

            I = self._fold(I)
            Q = self._fold(Q)
            U = self._fold(U)
            V = self._fold(V)

        self.data = {'I': I, 'Q': Q, 'U': U, 'V': V}

    def _plot_1d(self, axis, stokes):
        
        fig, ax = plt.subplots(figsize=(7, 5))

        # Average along opposite axis
        avg_axis = int(not axis)

        bins = self.data['I'].shape[axis]

        # Set plot parameters for either lightcurve or 1D spectrum
        if axis == 0:
            plottype = 'lc'
            valmin, valmax = (-0.5, 0.5) if self.fold else (self.tmin, self.tmax)
            ax.set_xlabel('Time (hours)')
        else:
            plottype = 'spectrum'
            valmin, valmax = self.fmin, self.fmax
            ax.set_xlabel('Frequency (MHz)')

        interval = (valmax - valmin) / bins
        x = np.array([valmin + i*interval for i in range(bins)])

        for pol in stokes:

            # Catch RuntimeWarning that occurs when averaging empty time/freq slices
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y = np.nanmean(self.data[pol], axis=avg_axis)

            ax.plot(x, y.real, label=pol)

            # Record to csv
            if axis == 0:
                label = 'time'
                values = self.time_start + x * u.hour
            else:
                label = 'frequency'
                values = self.fmin * x * u.MHz
            
            df = pd.DataFrame({
                label: values,
                'flux_density': y.real.reshape(1, -1)[0]
            })
            df.dropna().to_csv(
                f'{self.ds_path}/{self.src.lower()}_{plottype}_stokes{pol.lower()}.csv'
            )


        # Calculate rms from imaginary Stokes I data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            v = np.nanmean(self.data['I'], axis=avg_axis)
            rms = np.nanstd(v.imag)

        ax.axhline(rms, ls=':', color='r', label=f'rms={rms:.1f} mJy')
        ax.axhline(-rms, ls=':', color='r')

        ax.legend()
        
        ax.set_ylabel('Flux Density (mJy)')

        if self.save:
            fig.savefig(
                f'{self.ds_path}/{self.src.lower()}_{plottype}_{bins}bins.png',
                bbox_inches='tight',
                format='png',
                dpi=300,
            )

        return fig, ax

    def rebin(self, o, n, axis):
        '''Create unitary array compression matrix from o -> n length.

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
        '''

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
        '''Re-bin along time / frequency axes conserving flux.'''

        if new_shape == array.shape:
            return array

        if new_shape[0] > array.shape[0] or new_shape[1] > array.shape[1]:
            raise ValueError('New shape should not be greater than old shape in either dimension')

        time_comp = self.rebin(array.shape[0], new_shape[0], axis=0)
        freq_comp = self.rebin(array.shape[1], new_shape[1], axis=1)
        result = time_comp @ np.array(array) @ freq_comp

        return result

    def plot_crosspol_ds(self, cmax=20, cmin=0):

        U = self.data['U']
        V = self.data['V']
        data = np.sqrt(np.abs(U**2 + V**2))

        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.2))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            np.transpose(data),
            extent=[self.tmin, self.tmax, self.fmin, self.fmax],
            aspect='auto',
            origin='lower',
            norm=norm,
            clim=(cmin, cmax),
            cmap='plasma',
        )
        ax.text(
            0.05,
            0.95,
            r'sqrt(U^2 + V^2)',
            color='white',
            weight='heavy',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')],
            transform=ax.transAxes,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label('Flux Density (mJy)')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Frequency (MHz)')

        if self.save:
            data.dump(f'{self.ds_path}/{self.src.lower()}_crosspol_ds.npy')

            path_template = '{}/{}_crosspol_subbed_ds_favg{}-tavg{}.png'
            fig.savefig(
                path_template.format(self.ds_path, self.src.lower(), self.favg, self.tavg),
                bbox_inches='tight',
                format='png',
                dpi=300,
            )

        return fig, ax

    def plot_lightcurve(self, stokes):

        lc_fig, lc_ax = self._plot_1d(axis=0, stokes=stokes)

        return lc_fig, lc_ax

    def plot_spectrum(self, stokes):

        sp_fig, sp_ax = self._plot_1d(axis=1, stokes=stokes)

        return sp_fig, sp_ax

    def plot_ds(self, stokes, cmax=20, imag=False):

        data = self.data[stokes].imag if imag else self.data[stokes].real

        tmin, tmax = (-0.5, 0.5) if self.fold else (self.tmin, self.tmax)
        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.2))
        cmin = -2 if stokes == 'I' else -cmax
        cmax = cmax

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            np.transpose(data),
            extent=[tmin, tmax, self.fmin, self.fmax],
            aspect='auto',
            origin='lower',
            norm=norm,
            clim=(cmin, cmax),
            cmap='plasma',
        )
        ax.text(
            0.05,
            0.95,
            f'Stokes {stokes}',
            color='white',
            weight='heavy',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')],
            transform=ax.transAxes,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label('Flux Density (mJy)')

        xlabel = 'Phase' if self.fold else 'Time (hours)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency (MHz)')

        if self.save:
            data.dump(f'{self.ds_path}/{self.src.lower()}_{stokes.lower()}_ds.npy')

            path_template = '{}/{}_stokes{}_subbed_ds_favg{}-tavg{}.png'
            fig.savefig(
                path_template.format(self.ds_path, self.src.lower(), stokes.lower(), self.favg, self.tavg),
                bbox_inches='tight',
                format='png',
                dpi=300,
            )

        return fig, ax

    def plot_pol_ds(self, cmax=20):

        Idata = self.data['I'].real
        Vdata = self.data['V'].real

        rms = np.nanstd(self.data['V'].imag) / 40
        data = Vdata / Idata
        data[Idata < 3*rms] = np.nan

        tmin, tmax = (-0.5, 0.5) if self.fold else (self.tmin, self.tmax)
        norm = ImageNormalize(data, interval=ZScaleInterval(contrast=0.2))

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            np.transpose(data),
            extent=[tmin, tmax, self.fmin, self.fmax],
            aspect='auto',
            origin='lower',
            norm=norm,
            clim=(-1, 1),
            cmap='coolwarm',
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label('Fractional Circular Polarisation')

        xlabel = 'Phase' if self.fold else 'Time (hours)'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency (MHz)')

        # if self.save:
        #     data.dump(f'{self.ds_path}/{self.src.lower()}_{stokes.lower()}_ds.npy')

        #     path_template = '{}/{}_stokes{}_subbed_ds_favg{}-tavg{}.png'
        #     fig.savefig(
        #         path_template.format(self.ds_path, self.src.lower(), stokes.lower(), self.favg, self.tavg),
        #         bbox_inches='tight',
        #         format='png',
        #         dpi=300,
        #     )

        return fig, ax

    def plot_acf(self, stokes='I', contrast=0.4):

        # Compute auto-correlation and select upper-right quadrant
        acf2d = correlate(self.data[stokes].real, self.data[stokes].real)
        acf2d = acf2d[acf2d.shape[0]//2:, acf2d.shape[1]//2:]

        # Reorder time-frequency axes and normalise
        acf2d = np.flip(acf2d, axis=1).T
        acf2d /= acf2d.max()

        # Plot 2D ACF
        acf_fig, acf_ax = plt.subplots(figsize=(7, 5))

        norm = ImageNormalize(acf2d, interval=ZScaleInterval(contrast=contrast))
        im = acf_ax.imshow(
            acf2d,
            extent=[0, self.tmax-self.tmin, 0, self.fmax-self.fmin],
            aspect='auto',
            norm=norm,
            cmap='plasma',
        )
        cb = acf_fig.colorbar(
            im,
            ax=acf_ax,
            fraction=0.05,
            pad=0.02,
        )
        cb.formatter.set_powerlimits((0, 0))
        cb.set_label('ACF')

        acf_ax.set_xlabel('Time Lag (h)')
        acf_ax.set_ylabel('Frequency Lag (MHz)')

        # Plot zero frequency lag trace
        acfz_fig, acfz_ax = plt.subplots(figsize=(7, 5))

        zero_trace_acf = acf2d[-1, 1:]
        time_lag = np.linspace(0, self.tmax-self.tmin, len(zero_trace_acf))
        acfz_ax.plot(
            time_lag,
            zero_trace_acf,
            color='k',
        )

        acfz_ax.set_xlabel('Time Lag (h)')
        acfz_ax.set_ylabel('ACF')

        if self.save:
            acf_fig.savefig(f'{self.ds_path}/{self.src.lower()}_acf.png')
            acfz_fig.savefig(f'{self.ds_path}/{self.src.lower()}_f0_acf.png')

        return acf_fig, acf_ax, acfz_fig, acfz_ax

