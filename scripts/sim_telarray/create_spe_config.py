import argparse
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from targetpipe.fitting.spe_sipm import pe_signal
from IPython import embed


def main():
    description = 'Create a SPE spectrum file for sim_telarray'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output', dest='output_path', action='store',
                        required=True, help='path to save the config file')
    args = parser.parse_args()

    x = np.linspace(-1, 11, 1101)
    k = np.arange(1, 11)
    params = dict(
        norm=1,
        eped=0,
        eped_sigma=0,
        spe=1.3,
        spe_sigma=0.1,
        lambda_=1,
        opct=0.6,
        pap=0.3,
        dap=0.4
    )
    pe_s = pe_signal(k[:, None], x[None, :], **params).sum(0)

    np.savetxt(args.output_path, np.column_stack([x, pe_s, pe_s]), delimiter='\t', fmt=['%8.6f', '%-12.5g', '%12.5g'])
    exe = "/Users/Jason/Software/outputs/sim_telarray_configs/spe/norm_spe "
    cmd = exe + args.output_path + " > " + args.output_path + ".temp"
    print(cmd)
    subprocess.call(cmd, shell=True)
    cmd = "mv {} {}".format(args.output_path + ".temp", args.output_path)
    subprocess.call(cmd, shell=True)

    x_n, pe_s_n, _ = np.loadtxt(args.output_path, delimiter='\t', unpack=True)

    print(np.average(x, weights=pe_s))
    print(np.average(x_n, weights=pe_s_n))

    # embed()

    plt.semilogy(x, pe_s)
    plt.semilogy(x_n, pe_s_n)
    # plt.plot(x, pe_s)
    # plt.plot(x_n, pe_s_n)
    plt.show()

if __name__ == '__main__':
    main()