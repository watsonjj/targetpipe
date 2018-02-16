import argparse
import numpy as np
from targetpipe.plots.quick_camera import plot_quick_camera
from matplotlib import pyplot as plt


def main():
    description = 'Compare the rms image between two rms_runavg numpy files'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--on', dest='on_file',
                        action='store', required=True,
                        help='rms_runavg numpy file for the "on" measurement')
    parser.add_argument('--off', dest='off_file',
                        action='store', required=True,
                        help='rms_runavg numpy file for the "off" measurement')

    args = parser.parse_args()

    on_rms_runavg = np.load(args.on_file)
    off_rms_runavg = np.load(args.off_file)

    onoff = on_rms_runavg - off_rms_runavg

    plot_quick_camera(onoff)
    plt.show()

    print("FINISHED")


if __name__ == '__main__':
    main()
