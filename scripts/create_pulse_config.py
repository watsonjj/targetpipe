#!python

"""
Create a pulse config file to be used as the input to sim_telarray
"""

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt


def pulse0(x, mean, variance):
    return (1. / np.sqrt(2 * np.pi * variance)) * \
           np.exp(-np.power((x - mean), 2) / (2 * variance))


def pulse1(x, pos):
    pulse = np.zeros(x.size)
    pulse[int(pos)] = 1
    return pulse


def pulseln(x, sigma, mu):
    x[x <= 0] = np.nan
    p = (1 / x * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.power(np.log(x) - mu, 2)/(2 * np.power(sigma, 2))))
    return p


def pulse_sipm(x, sigma, mu, amp, mean, variance):
    return pulseln(x, sigma, mu) - amp * pulse0(x, mean, variance)


def print_disc(array, output_dir, name):
    path = os.path.join(output_dir, 'disc_{}.dat'.format(name))

    try:
        os.remove(path)
    except OSError:
        pass
    file = open(path, "x")

    print("#### Input signal pulse shape for pre-sum / discriminator",
          file=file)
    print("#### (for now identical to FADC pulse shape but different format)",
          file=file)
    print("#", file=file)
    time = 0.0
    for val in array:
        print("{0:.1f}  {1:.6f}".format(time, val), file=file)
        time += 0.5
    print("", file=file)

    file.close()


def print_pulse(array, output_dir, name):
    path = os.path.join(output_dir, 'pulse_{}.dat'.format(name))

    try:
        os.remove(path)
    except OSError:
        pass
    file = open(path, "x")

    print("#### FADC pulse shape for a 0.25 GHz system:", file=file)
    print("####", file=file)
    print("#", file=file)
    print("# T=0.5  !Time step is 0.5 nanoseconds", file=file)
    print("#", file=file)
    for val in array:
        print("{0:.6f}  {1:.6f}".format(val, val), file=file)
    print("", file=file)

    file.close()


def main():
    parser = argparse.ArgumentParser(description='Create a pulse config file '
                                                 'to be used as the input to '
                                                 'sim_telarray')
    parser.add_argument('-o', '--output', dest='output_dir', action='store',
                        required=True, help='directory to store the '
                                            'output files')
    parser.add_argument('-n', '--name', dest='name', action='store',
                        required=True, help='name for the output files')

    args = parser.parse_args()

    x = np.arange(0, 25.5, 0.5)
    # pulse = pulse0(x, 7, 5)
    # pulse = pulse1(x, 7)
    # pulse = pulseln(x, 0.32, 2)
    pulse = pulse_sipm(x, 0.32, 2, 0.4, 13, 6)
    norm = pulse / np.nanmax(pulse)

    out = np.nan_to_num(norm)

    print_disc(out, args.output_dir, args.name)
    print_pulse(out, args.output_dir, args.name)

    print(out)
    plt.plot(x, out)
    plt.show()


if __name__ == '__main__':
    main()
