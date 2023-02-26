import click
import logging
import matplotlib.pyplot as plt
from dstools.utils import setupLogger
from dstools.dynamic_spectrum import DynamicSpectrum

logger = logging.getLogger(__name__)

@click.command()
@click.option('-f', '--favg', default=1, type=int,
              help='Averaging factor across frequency axis..')
@click.option('-t', '--tavg', default=1, type=int,
              help='Averaging factor across time axis.')
@click.option('-I', '--cmax_i', default=50, type=float,
              help='Maximum colormap normalisation in Stokes I.')
@click.option('-Q', '--cmax_qu', default=10, type=float,
              help='Maximum colormap normalisation in Stokes Q and U.')
@click.option('-V', '--cmax_v', default=50, type=float,
              help='Maximum colormap normalisation in Stokes V.')
@click.option('-r', '--real', is_flag=True, default=True,
              help='Toggle plotting of real component of visibilities in dynamic spectra.')
@click.option('-i', '--imag', is_flag=True, default=False,
              help='Toggle plotting of imaginary component of visibilities in dynamic spectra.')
@click.option('-s', '--stokes', default='IQUV',
              help='Stokes parameters that will be included in each plot.')
@click.option('-d', '--dspec', is_flag=True, default=False,
              help='Plot dynamic spectrum.')
@click.option('-c', '--pol_dspec', is_flag=True, default=False,
              help='Plot fractional polarisation dynamic spectrum.')
@click.option('-l', '--lightcurve', is_flag=True, default=False,
              help='Plot channel-averaged lightcurve.')
@click.option('-p', '--spectrum', is_flag=True, default=False,
              help='Plot time-averaged spectrum.')
@click.option('-x', '--crosspols', is_flag=True, default=False,
              help='Plot quadrature sum of cross-polarisations as a diagnostic for RFI and leakage.')
@click.option('-a', '--acf', is_flag=True, default=False,
              help='Plot 2D auto-correlation function.')
@click.option('-R', '--trim', is_flag=True, default=True,
              help='Remove flagged channels at top/bottom of band.')
@click.option('-F', '--fold', is_flag=True, default=False,
              help='Toggle to enable folding of data. Must also provide period with -T.')
@click.option('-T', '--period', default=None, type=float,
              help='Period to use when folding data.')
@click.option('-o', '--period_offset', default=0, type=float,
              help='Period phase offset to use when folding data.')
@click.option('-C', '--calscans', is_flag=True, default=True,
              help='Toggle inclusion of null-valued time chunks while off-source (e.g. calibrator scans, wind stows)')
@click.option('-B', '--band', default='L', type=click.Choice(['low', 'mid', 'L', 'C', 'X']),
              help='Frequency band. Must correspond to a sub-directory of <project>/dynamic_spectra/')
@click.option('-S', '--save', is_flag=True, default=False,
              help='Toggle saving of plots and lightcurve data.')
@click.option('-N', '--versionname', default=None,
              help='Prefix for different processing versions')
@click.option('-v', '--verbose', is_flag=True, default=False)
@click.argument('project')
def main(
    favg,
    tavg,
    cmax_i,
    cmax_qu,
    cmax_v,
    real,
    imag,
    stokes,
    dspec,
    pol_dspec,
    lightcurve,
    spectrum,
    crosspols,
    acf,
    fold,
    trim,
    period,
    period_offset,
    calscans,
    band,
    save,
    versionname,
    verbose,
    project,
):

    setupLogger(verbose)

    prefix = f'{versionname}_' if versionname else ''

    cmax = {
        'I': cmax_i,
        'Q': cmax_qu,
        'U': cmax_qu,
        'V': cmax_v,
    }

    ds = DynamicSpectrum(
        project=project,
        band=band,
        calscans=calscans,
        tavg=tavg,
        favg=favg,
        prefix=prefix,
        fold=fold,
        trim=trim,
        period=period,
        period_offset=period_offset,
        save=save,
    )

    # Dynamic Spectrum
    # --------------------------------------
    if dspec:
        for s in stokes:
            if real:
                ds.plot_ds(stokes=s, cmax=cmax[s])
            if imag:
                ds.plot_ds(stokes=s, cmax=cmax[s], imag=True)

    if pol_dspec:
        ds.plot_pol_ds(cmax=cmax['I'])

    # Cross-polarisations sqrt(U^2 + V^2)
    if crosspols:
        ds.plot_crosspol_ds(cmax=cmax['I'])

    # Spectrum
    # --------------------------------------
    if spectrum:
        ds.plot_spectrum(stokes=stokes)

    # Light Curve
    # --------------------------------------
    if lightcurve:
        ds.plot_lightcurve(stokes=stokes)

    # Dynamic Spectrum 2D Auto-correlation Function
    # --------------------------------------
    if acf:
        for s in stokes:
            ds.plot_acf(stokes=s, contrast=0.2)

    plt.show()


if __name__ == '__main__':
    main()
