import dstools
import colorlog
import logging
import astropy.units as u

from dataclasses import dataclass

CONFIGS = ['6km', '750_no6', '750_6', 'H168']
BANDS = ['AK_low', 'AK_mid', 'AT_L', 'AT_C', 'AT_X', 'MKT_UHF', 'MKT_L']

def parse_casa_args(func, module, kwargs, args=None):
    path = dstools.__path__[0]
    path = f'{path}/cli/{module}'

    if args is not None:
        args = [f'{kwargs.pop(arg)}' for arg in args]
        argstr = ' '.join(args)
    else:
        argstr = ''

    kw_args = []
    for key, val in kwargs.items():

        if isinstance(val, bool):
            boolval = '' if val else 'no-'
            flag = f'--{boolval}{key}'
        elif val == None or val == '':
            continue
        else:
            flag = f'--{key} {val}'

        kw_args.append(flag)
    
    kwargstr = ' '.join(kw_args)

    return path, argstr, kwargstr

def colored(msg):

    return '\033[91m{}\033[0m'.format(msg)


def prompt(msg):

    msg = msg[:-1] + ' (y/n)?\n'

    resp = input(colored(msg))
    if resp not in ['y', 'n']:
        resp = input(colored(msg))

    return True if resp == 'y' else False


def update_param(name, val, dtype):
    while True:
        newval = input('Enter new {} (currently {}): '.format(name, val))

        # Accept old value if nothing entered
        if not newval:
            break

        # Test for correct input type
        try:
            val = dtype(newval)
            break
        except ValueError:
            print('{} must be of type {}'.format(name, type(val)))

    return val

@dataclass
class Array:

    band: str='AT_L'
    config: str='6km'

    def __post_init__(self):
        
        telescope = self.band.split('_')[0]
        self.config = self.config if telescope == 'AT' else telescope

        # Wavelengths are taken at top of the band to calculate resolution
        wavelengths = {
            'AK_low': 0.4026,  
            'AK_mid': 0.2450,
            'AK_high': -1,
            'AT_L': 0.0967,
            'AT_C': 0.0461,
            'AT_X': 0.0300,
            'MKT_UHF': 0.2954,
            'MKT_L': 0.1795,
        }
        max_baselines = {
            'MKT': 8000,
            'AK': 6000,
            '6km': 6000,
            '750_no6': 750,
            '750_6': 5020,
            'H168': 185,
        }

        # Frequencies are taken at centre of band for Taylor expansion
        frequencies = {
            'AK_low': '888.49',
            'AK_mid': '1367.49',
            'AK_high': '',
            'AT_L': '2100',
            'AT_C': '5500',
            'AT_X': '9000',
            'MKT_UHF': '797.5',
            'MKT_L': '1285',
        }

        # Primary beam taken as half-power at bottom of the band,
        # though ATCA C/X band are narrower as the source density is low
        primary_beams = {
            'AK_low': 1.111,
            'AK_mid': 0.84,
            'AK_high': -1,
            'AT_L': 0.825,
            'AT_C': 0.2225,
            'AT_X': 0.1460,
            'MKT_UHF': 2.56,
            'MKT_L': 2.56,
        }

        cellsize = {
            'AK_low': 2.5,
            'AK_mid': 1.5,
            'AK_high': -1,
            'AT_L': 0.66,
            'AT_C': 0.32,
            'AT_X': 0.21,
            'MKT_UHF': 1.5,
            'MKT_L': 1,
        }
        
       
        self.frequency = frequencies[self.band]

        # resolution = wavelengths[self.band] /  max_baselines[self.config] * u.rad.to(u.arcsec)
        self.cell = cellsize[self.band]

        self.imradius = primary_beams[self.band]
        self.imsize = int(round(self.imradius / (self.cell*u.arcsec.to(u.deg)), 0))

    def __str__(self):

        return str({
            'band': self.band,
            'config': self.config,
            'frequency': self.frequency,
            'cell': self.cell,
            'imradius': self.imradius,
            'imsize': self.imsize,
        })

    
def setupLogger(verbose, filename=None):

    level = logging.DEBUG if verbose else logging.INFO

    # Get root logger disable any existing handlers, and set level
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(level)

    # Turn off some bothersome verbose logging modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    if filename:
        formatter = logging.Formatter(
            '%(levelname)-8s %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    colorformatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(asctime)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        })

    stream_handler = colorlog.StreamHandler()
    stream_handler.setFormatter(colorformatter)

    root_logger.addHandler(stream_handler)

    return None
