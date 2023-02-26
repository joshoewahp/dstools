import click
import logging
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from dstools.utils import setupLogger
from dstools.dynamic_spectrum import DynamicSpectrum
from astropy.time import Time

logger = logging.getLogger(__name__)

@click.command()
@click.argument('data')
@click.argument('period')
@click.argument('forecast_date')
def main(data, period, forecast_date):

    fig, ax = plt.subplots()

    data = pd.read_csv(data)

    pulse = data[data.flux_density.abs() == data.flux_density.abs().max()].iloc[0]
    
    times = [Time(t).mjd for t in data.time.values]
    fluxes = list(data.flux_density.values)
    ax.plot(times, fluxes, color='k')
    ax.scatter(Time(pulse.time).mjd, pulse.flux_density, marker='*', color='r')

    print(pulse.time)

    time = Time(Time(pulse.time) + 0*u.hour)
    forecast_date = Time(forecast_date)
    candidate_times = []

    while time < forecast_date:
        candidate_times.append(time)
        time = Time(time + float(period)*u.hour)
        
    plt.show()

    for time in candidate_times:
        print(time.value)



if __name__ == '__main__':
    main()
