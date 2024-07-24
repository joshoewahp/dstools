import time
import click
import h5py
import logging
import os
import warnings
import itertools as it
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astroutils.logger import setupLogger
from casacore.tables import table
from concurrent.futures import ProcessPoolExecutor, wait


warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

logger = logging.getLogger(__name__)


def get_pb_correction(position, primary_beam):

    with fits.open(primary_beam) as hdul:
        header, data = hdul[0].header, hdul[0].data
        data = data[0, 0, :, :]

    wcs = WCS(header, naxis=2)
    x, y = wcs.wcs_world2pix(position.ra, position.dec, 1)
    x, y = int(x // 1), int(y // 1)
    xmax, ymax = data.shape

    # Check position is within limits of supplied PB image
    im_outside_limit = [
        x < 0,
        x > xmax,
        y < 0,
        y > ymax,
    ]
    if any(im_outside_limit):
        logger.warning(
            f"Position {c} outside of supplied PB image, disabling PB correction."
        )
        return 1

    scale = data[x, y]

    logger.debug(
        f"PB correction scale {scale:.2f} measured at pixel {x},{y} in image of size {xmax},{ymax}"
    )

    return scale


def process_baseline(ms, times, baseline, datacolumn):

    i, (ant1, ant2) = baseline

    tab = table(ms, ack=False, lockoptions="autonoread")
    bl_tab = tab.query("(ANTENNA1==$ant1) && (ANTENNA2==$ant2)")

    # Identify missing integrations on this baseline
    bl_time = bl_tab.getcol("TIME")
    missing_times = [t for t in times if t not in bl_time]

    # Add back to time column and identify indices of good integrations
    bl_time = np.sort(np.append(bl_time, missing_times))
    data_idx = np.argwhere(~np.in1d(bl_time, missing_times)).ravel()

    # Calculate UVrange for each baseline
    bl_uvw = bl_tab.getcol("UVW")
    bl_uvdist = np.sqrt(np.sum(np.square(bl_uvw), axis=1))

    data = {
        "baseline": i,
        "data_idx": data_idx,
        "data": bl_tab.getcol(datacolumn),
        "flags": bl_tab.getcol("FLAG"),
        "uvdist": np.nanmean(bl_uvdist, axis=0),
    }

    tab.close()
    bl_tab.close()

    return data


@click.command()
@click.option(
    "-F",
    "--noflag",
    is_flag=True,
    default=False,
    help="Remove flagging mask.",
)
@click.option(
    "-B",
    "--baseline-average",
    is_flag=True,
    default=False,
    help="Average date over baseline axis.",
)
@click.option(
    "-u",
    "--minuvdist",
    type=float,
    default=0,
    help="Minimum UV distance in meters to retain if averaging over baseline axis.",
)
@click.option(
    "-d",
    "--datacolumn",
    type=click.Choice(["data", "corrected", "model"]),
    default="data",
    help="Selection of DATA, CORRECTED_DATA, or MODEL column.",
)
@click.option(
    "-p",
    "--phasecentre",
    type=str,
    nargs=2,
    default=None,
    help="RA and Dec of phasecentre at which to extract DS.",
)
@click.option(
    "-P",
    "--primary-beam",
    type=click.Path(),
    default=None,
    help="Path to primary beam image with which to correct flux scale. Must also provide phasecentre.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
@click.argument("ms")
@click.argument("outfile")
def main(
    noflag,
    baseline_average,
    minuvdist,
    datacolumn,
    phasecentre,
    primary_beam,
    verbose,
    ms,
    outfile,
):

    setupLogger(verbose=verbose)

    # Check that selected column exists in MS
    columns = {
        "data": "DATA",
        "corrected": "CORRECTED_DATA",
        "model": "MODEL_DATA",
    }
    datacolumn = columns[datacolumn]

    tab = table(ms, ack=False, lockoptions="autonoread")
    if not datacolumn in tab.colnames():
        logger.error(f"{datacolumn} column does not exist in {ms}")
        exit(1)
    tab.close()

    scale = 1
    if phasecentre is not None:

        # Parse phasecentre
        ra, dec = phasecentre
        ra_unit = "hourangle" if ":" in ra or "h" in ra else "deg"
        position = SkyCoord(ra=ra, dec=dec, unit=(ra_unit, "deg"))
        c = position.to_string(style="hmsdms", precision=3)
        logger.debug(f"Rotating phasecentre to {c}")

        # Apply phasecentre rotation
        os.system(f"dstools-rotate -- {ms} {c} 1>/dev/null")

        # Use rotated MeasurementSet for subsequent processing
        ms = ms.replace(".ms", ".rotated.target.ms")

        # Correct flux scale for primary beam
        if primary_beam is not None:
            scale = get_pb_correction(position, primary_beam)

    # Get antenna count and time / frequency arrays
    tab = table(ms, ack=False, lockoptions="autonoread")

    # Throw away autocorrelations
    tab = tab.query("ANTENNA1 != ANTENNA2")

    # Get time, frequency, and baseline axes
    times = np.unique(tab.getcol("TIME"))
    tf = table(f"{ms}::SPECTRAL_WINDOW", ack=False)
    freqs = tf[0]["CHAN_FREQ"]

    antennas = np.unique(
        np.append(
            tab.getcol("ANTENNA1"),
            tab.getcol("ANTENNA2"),
        ),
    )

    tf.close()
    tab.close()

    # Calculate number of baselines and initialise output data arrays
    nbaselines = len(antennas) * (len(antennas) - 1) // 2
    data_shape = (nbaselines, len(times), len(freqs), 4)

    waterfall = np.full(data_shape, np.nan, dtype=complex)
    flags = np.full(data_shape, np.nan, dtype=bool)
    uvdist = np.full(nbaselines, np.nan)

    logger.debug(
        f"Extracting dynamic spectrum along:\n  {nbaselines} baselines\n  {len(times)} integrations\n  {len(freqs)} channels\n  4 polarisations"
    )

    # Construct 4D data and flag cubes on each baseline separately
    # to verify indices of missing data (e.g. due to correlator dropouts)
    with ProcessPoolExecutor(max_workers=15) as executor:
        processes = [
            executor.submit(
                process_baseline,
                ms,
                np.copy(times),
                baseline,
                datacolumn,
            )
            for baseline in enumerate(it.combinations(antennas, 2))
        ]
        wait(processes)

    # Insert data into 4D data / flag cubes
    results = [p.result() for p in processes]
    for baseline in results:
        baseline_idx, data_idx = baseline["baseline"], baseline["data_idx"]
        waterfall[baseline_idx, data_idx] = baseline["data"]
        flags[baseline_idx, data_idx] = baseline["flags"]
        uvdist[baseline_idx] = baseline["uvdist"]

    # Apply flags
    if not noflag:
        waterfall[flags] = np.nan

    # Optionally average over baseline axis.
    # Retain 4D cube with length 1 baseline axis for data structure uniformity
    if baseline_average:
        logger.debug(f"Averaging over baseline axis with uvdist > {minuvdist}m")
        blmask = uvdist > minuvdist
        waterfall = waterfall[blmask, :, :, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            waterfall = np.expand_dims(np.nanmean(waterfall, axis=0), axis=0)
            uvdist = np.expand_dims(np.nanmean(uvdist, axis=0), axis=0)

    # Apply primary beam correction
    waterfall /= scale

    header = {
        "baselines": nbaselines,
        "channels": len(freqs),
        "integrations": len(times),
        "correlations": nbaselines * len(freqs) * len(times) * 4,
    }

    # Write all data to file
    with h5py.File(outfile, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("frequency", data=freqs)
        f.create_dataset("uvdist", data=uvdist)
        f.create_dataset("flux", data=waterfall)

    # Clean up intermediate files
    # if "rotated" in ms:
    #     os.system(f"rm -r {ms} 2>/dev/null")
    os.system("rm *.log *.last 2>/dev/null")


if __name__ == "__main__":
    main()
