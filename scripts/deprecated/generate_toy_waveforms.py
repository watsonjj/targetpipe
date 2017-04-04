"""
Produce a numpy file containing fake "toy" waveforms that can be read by toyio
"""

import numpy as np
import argparse
from astropy import log
from targetpipe.io.toy_waveforms import ToyWaveformsCHECM as ToyWaveforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        required=True,
                        help='Path for saving the output numpy file.')
    parser.add_argument('-E', '--n_events', dest='n_events',
                        action='store', required=False, default=10, type=int,
                        help='Number of events to simulate. Ignored if '
                             'fill_factor is set.')
    parser.add_argument('-F', '--fill_factor', dest='fill_factor',
                        action='store', required=False, default=None, type=int,
                        help='Fill all cells at least fill_factor times.')
    parser.add_argument('-P', '--toy_pulses', dest='toy_pulses',
                        action='store_true', required=False, default=False,
                        help='Produce pulses in the toy model of the '
                             'waveforms.')

    logger_detail = parser.add_mutually_exclusive_group()
    logger_detail.add_argument('-q', '--quiet', dest='quiet',
                               action='store_true', default=False,
                               help='Quiet mode')
    logger_detail.add_argument('-v', '--verbose', dest='verbose',
                               action='store_true', default=False,
                               help='Verbose mode')
    logger_detail.add_argument('-d', '--debug', dest='debug',
                               action='store_true', default=False,
                               help='Debug mode')

    args = parser.parse_args()

    if args.quiet:
        log.setLevel(40)
    if args.verbose:
        log.setLevel(20)
    if args.debug:
        log.setLevel(10)

    log.info("[SCRIPT] generate_toy_waveforms")

    tw = ToyWaveforms(args.n_events)
    if args.fill_factor:
        tw.init_all_cells_filled(args.fill_factor)

    if not args.toy_pulses:
        wf = tw.get_base_wf()
    else:
        wf = tw.get_pulse_wf()

    log.info("Saving toy-waveforms to: {}".format(args.output_path))
    np.save(args.output_path, wf)

if __name__ == '__main__':
    main()
