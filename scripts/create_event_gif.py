#!python
"""
Create a animated gif of an waveforms, similar to those produced in libCHEC for
the inauguration press release.
"""

import argparse
from astropy import log

from targetpipe.io.eventfilereader import TargetioFileReader
from ctapipe.plotting.camera import CameraPlotter

from matplotlib import pyplot as plt
from matplotlib import animation

import numpy as np
from tqdm import tqdm
import os


def main():
    parser = argparse.ArgumentParser(description='Create a gif of an waveforms')
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the input file')
    parser.add_argument('-e', '--event', dest='event_req', action='store',
                        required=True, type=int,
                        help='event index to plot (not id!)')
    parser.add_argument('--id', dest='event_id_f', action='store_true',
                        default=False, help='-e will specify event_id instead '
                                            'of index')
    parser.add_argument('-T', '--time', dest='time', action='store',
                                required=True, nargs=2, type=int,
                                help='time window to generate gif between e.g.'
                                     ' "-T 20 21"')
    parser.add_argument('-P', '--pixels', dest='pixels', action='store',
                        required=True, nargs=2, type=int,
                        help='two pixels to draw the waveform of e.g. '
                             '"-P 10 221"')
    parser.add_argument('--maxpix', dest='maxpix_f',
                                action='store_true', default=False,
                                help='output a png of the timeslice at max '
                                     'amplitude value, with annotated '
                                     'pixel_ids. Useful for identifying where '
                                     'to create gif. -P and -T are overwritten'
                                     'in this mode.')
    parser.add_argument('-t', '--telescope', dest='tel', action='store',
                        default=-1, help='telecope to view (default = first)')
    parser.add_argument('-c', '--channel', dest='channel', action='store',
                        default=0, help='channel to view (default = first)')



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

    log.info("[SCRIPT] create_event_gif")

    log.debug("[file] Reading file")
    file_reader = TargetioFileReader(None, None, input_path = args.input_path)
    event = file_reader.get_event(args.event_req, args.event_id_f)

    # Calibrate event

    # Gather waveforms/args values
    event_id = event.dl0.event_id
    tel_list = list(event.dl0.tels_with_data)
    tel = tel_list[0] if args.tel == -1 else args.tel
    channel = args.channel
    data = event.r0.tel[tel].adc_samples[channel]
    n_pixels = event.inst.num_pixels[tel]
    t0 = args.time[0] if args.time[0] > 0 else 0
    t1 = args.time[1] if args.time[1] < np.size(data[0, :]) \
        else np.size(data[0, :])

    # Print waveforms/args values
    log.info("[event_id] {}".format(event_id))
    log.info("[event_index] {}".format(event.count))
    log.info("[telescope] {}".format(tel))
    log.info("[channel] {}".format(channel))

    # Draw figures
    fig = plt.figure(figsize=(24, 10))
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 3)
    ax2 = fig.add_subplot(1, 2, 2)

    plotter = CameraPlotter(event)
    waveform0 = plotter.draw_waveform(data[args.pixels[0], :], ax0)
    ax0.set_title("Pixel: {}".format(args.pixels[0]))
    line0 = plotter.draw_waveform_positionline(0, ax0)
    waveform1 = plotter.draw_waveform(data[args.pixels[1], :], ax1)
    ax1.set_title("Pixel: {}".format(args.pixels[1]))
    line1 = plotter.draw_waveform_positionline(0, ax1)

    camera = plotter.draw_camera(tel, data[:, 0], ax2)
    ax2.set_title("[Input] {} [Event] {} [Telescope] {} [Channel] {} "
                  "[Time] {}-{} UNCALIBRATED".format(file_reader.filename,
                                                     event_id, tel,
                                                     channel, t0, t1))

    if args.maxpix_f:
        # Create image of maxpix
        flatten_sorted = data.flatten().argsort()[:][::-1]
        pixels_max = np.unravel_index(flatten_sorted, data.shape)[0]
        pixels_max_sortedunique, ind = np.unique(pixels_max, return_index=True)
        pixels_max_unique = pixels_max[ind[np.argsort(ind)]]
        p0 = pixels_max_unique[0]
        p1 = pixels_max_unique[8]
        t0 = np.unravel_index(np.argmax(data), data.shape)[1]

        ax0.cla()
        plotter.draw_waveform(data[p0, :], ax0)
        ax0.set_title("Pixel: {}".format(p0))
        ax1.cla()
        plotter.draw_waveform(data[p1, :], ax1)
        ax1.set_title("Pixel: {}".format(p1))
        camera.image = data[:, t0]
        ax2.set_title("[Input] {} [Event] {} [Telescope] {} [Channel] {} "
                      "[Time] {} UNCALIBRATED".format(file_reader.filename, event_id, tel,
                                            channel, t0, t1))
        plotter.draw_camera_pixel_annotation(tel, p0, p1, ax2)
        plotter.draw_camera_pixel_ids(tel, np.arange(n_pixels), ax2)

        output_name = "{}_e{}_t{}_c{}_maxpix.pdf".format(file_reader.filename,
                                                         event_id, tel,
                                                         channel)
        output_path = os.path.join(file_reader.output_directory, output_name)
        if not os.path.exists(os.path.dirname(output_path)):
            log.info("[output] Creating directory: {}".format(
                os.path.dirname(output_path)))
            os.makedirs(os.path.dirname(output_path))
        log.info("[output] {}".format(output_path))
        plt.savefig(output_path, format='pdf')

    else:
        plotter.draw_camera_pixel_annotation(tel, args.pixels[0],
                                             args.pixels[1], ax2)
        # Create animation
        div = 3
        increment = 1/div
        n_frames = int((t1 - t0)/increment)
        interval = int(500*increment)

        output_name = "{}_e{}_t{}_c{}_camera.gif".format(file_reader.filename,
                                                         event_id, tel, channel)
        output_path = os.path.join(file_reader.output_directory, output_name)
        if not os.path.exists(os.path.dirname(output_path)):
            log.info("[output] Creating directory: {}".format(
                os.path.dirname(output_path)))
            os.makedirs(os.path.dirname(output_path))
        log.info("[output][in_progress] {}".format(output_path))

        with tqdm(total=n_frames, desc="Creating animation", smoothing=0) \
                as pbar:
            def animate(i):
                pbar.update(1)
                camera.image = data[:, int(t0+(i//div))]
                line0.set_xdata(t0+(i/div))
                line1.set_xdata(t0+(i/div))
                return line1, line0

            anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                           interval=interval, blit=True)
            anim.save(output_path, writer='imagemagick')
    log.info("[COMPLETE]")

if __name__ == '__main__':
    main()
