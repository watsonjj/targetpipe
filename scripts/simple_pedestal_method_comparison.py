import numpy as np
from matplotlib import pyplot as plt
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from tqdm import tqdm
import argparse
from os import makedirs
from os.path import splitext, join, dirname, basename, exists
from IPython import embed

# CHEC-M
N_ROWS = 8
N_COLUMNS = 64
N_BLOCKS = N_ROWS * N_COLUMNS
N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
SKIP_SAMPLE = 32
SKIP_END_SAMPLE = 0
SKIP_EVENT = 1
SKIP_END_EVENT = 1

# OTHER
# N_ROWS = 8
# N_COLUMNS = 16
# N_BLOCKS = N_ROWS * N_COLUMNS
# N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
# SKIP_SAMPLE = 0
# SKIP_END_SAMPLE = 0
# SKIP_EVENT = 0
# SKIP_END_EVENT = 0


def get_bp_r_c(cells):
    blockphase = cells % N_BLOCKSAMPLES
    row = (cells // N_BLOCKSAMPLES) % 8
    column = (cells // N_BLOCKSAMPLES) // 8
    return blockphase, row, column


class Reader:
    def __init__(self, path):
        self.path = path

        self.reader = TIOReader(self.path, N_CELLS,
                                SKIP_SAMPLE, SKIP_END_SAMPLE,
                                SKIP_EVENT, SKIP_END_EVENT)
        self.n_events = self.reader.fNEvents
        self.run_id = self.reader.fRunID
        self.n_pix = self.reader.fNPixels
        self.n_modules = self.reader.fNModules
        self.n_tmpix = self.n_pix // self.n_modules
        self.n_samples = self.reader.fNSamples
        self.n_cells = self.reader.fNCells

        self.max_blocksinwf = self.n_samples // N_BLOCKSAMPLES + 1
        self.samples = np.zeros((self.n_pix, self.n_samples), dtype=np.uint16)
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

    def get_event(self, iev):
        self.reader.GetR0Event(iev, self.samples, self.first_cell_ids)

    def event_generator(self):
        for iev in range(self.n_events):
            self.get_event(iev)
            yield iev


def generate_pedestal(input_path, output_path):
    reader = Reader(input_path)
    n_modules = reader.n_modules
    n_tmpix = reader.n_tmpix
    n_pix = reader.n_pix
    n_samples = reader.n_samples
    max_blocksinwf = reader.max_blocksinwf
    samples = reader.samples
    fci = reader.first_cell_ids

    tm = np.arange(n_pix, dtype=np.uint16) // n_tmpix
    tmpix = np.arange(n_pix, dtype=np.uint16) % n_tmpix

    ped_basic = np.zeros((n_modules, n_tmpix, N_CELLS))
    ped_blocks4 = np.zeros((n_modules, n_tmpix, N_CELLS, max_blocksinwf))
    ped_blocks8 = np.zeros((n_modules, n_tmpix, N_CELLS, max_blocksinwf * 2))
    ped_adrian = np.zeros((n_modules, n_tmpix, N_BLOCKS,
                           max_blocksinwf * N_BLOCKSAMPLES))
    ped_correction = np.zeros((n_modules, n_tmpix, n_samples))

    hits_basic = np.zeros((n_modules, n_tmpix, N_CELLS))
    hits_blocks4 = np.zeros((n_modules, n_tmpix, N_CELLS, max_blocksinwf))
    hits_blocks8 = np.zeros((n_modules, n_tmpix, N_CELLS, max_blocksinwf * 2))
    hits_adrian = np.zeros((n_modules, n_tmpix, N_BLOCKS,
                            max_blocksinwf * N_BLOCKSAMPLES))
    hits_correction = np.zeros((n_modules, n_tmpix, n_samples))

    pix_i = np.arange(n_pix)
    tm_i = tm[:, None] * np.ones((n_pix, n_samples), dtype=np.int)
    tmpix_i = tmpix[:, None] * np.ones((n_pix, n_samples), dtype=np.int)

    tm_ix = tm_i[..., None]
    tmpix_ix = tmpix_i[..., None]

    sample_range = np.arange(n_samples, dtype=np.int)[None, :]
    sample_zeros = np.zeros(n_samples, dtype=np.int)[None, :]
    sample_index = sample_range + np.zeros(n_pix, dtype=np.int)[:, None]

    def mean(pedestal, hits, samp):
        return pedestal * (hits / (hits + 1)) + samp / (hits + 1)

    source = reader.event_generator()
    n_events = reader.n_events
    with tqdm(total=n_events) as pbar:
        for _ in source:
            pbar.update(1)

            uncorrected_cells = sample_range + fci[:, None]
            cells = uncorrected_cells % N_CELLS
            fci_cell = sample_zeros + fci[:, None]

            uncor_x = uncorrected_cells[:, :, None]
            cells_x = cells[:, :, None]
            fci_cell_x = fci_cell[:, :, None]

            i_block4 = uncor_x // N_BLOCKSAMPLES - \
                (uncor_x[:, 0, :] // N_BLOCKSAMPLES)[:, None]
            # i_block8 = uncor_x // (N_BLOCKSAMPLES // 2) - \
            #     (uncor_x[:, 0, :] // (N_BLOCKSAMPLES // 2))[:, None]
            bp_fci, r_fci, c_fci = get_bp_r_c(fci_cell_x)
            block_fci = r_fci + c_fci * N_ROWS
            block_pos = sample_range[:, :, None] + bp_fci

            s = samples[pix_i]
            s_x = s[..., None]

            # p = ped_basic[tm_i, tmpix_i, cells]
            # h = hits_basic[tm_i, tmpix_i, cells]
            # ped_basic[tm_i, tmpix_i, cells] = mean(p, h, s)
            # hits_basic[tm_i, tmpix_i, cells] += 1

            p = ped_blocks4[tm_ix, tmpix_ix, cells_x, i_block4]
            h = hits_blocks4[tm_ix, tmpix_ix, cells_x, i_block4]
            ped_blocks4[tm_ix, tmpix_ix, cells_x, i_block4] = mean(p, h, s_x)
            hits_blocks4[tm_ix, tmpix_ix, cells_x, i_block4] += 1

            # p = ped_blocks8[tm_ix, tmpix_ix, cells_x, i_block8]
            # h = hits_blocks8[tm_ix, tmpix_ix, cells_x, i_block8]
            # ped_blocks8[tm_ix, tmpix_ix, cells_x, i_block8] = mean(p, h, s_x)
            # hits_blocks8[tm_ix, tmpix_ix, cells_x, i_block8] += 1
            #
            p = ped_adrian[tm_ix, tmpix_ix, block_fci, block_pos]
            h = hits_adrian[tm_ix, tmpix_ix, block_fci, block_pos]
            ped_adrian[tm_ix, tmpix_ix, block_fci, block_pos] = mean(p, h, s_x)
            hits_adrian[tm_ix, tmpix_ix, block_fci, block_pos] += 1
            #
            # residual = s - ped_basic[tm_i, tmpix_i, cells]
            # p = ped_correction[tm_i, tmpix_i, sample_index]
            # h = hits_correction[tm_i, tmpix_i, sample_index]
            # ped_correction[tm_i, tmpix_i, sample_index] = mean(p, h, residual)
            # hits_correction[tm_i, tmpix_i, sample_index] += 1

    if not exists(dirname(output_path)):
        print("Creating directory: {}".format(dirname(output_path)))
        makedirs(dirname(output_path))

    np.savez(output_path,
             basic=ped_basic,
             blocks4=ped_blocks4,
             blocks8=ped_blocks8,
             adrian=ped_adrian,
             correction=ped_correction)

    print("Pedestals saved to: {}".format(output_path))


def apply_pedestal(input_path, pedestal_path):

    print(__name__)
    fig_dir = join(dirname(input_path), splitext(basename(input_path))[0],
                   "simple_pedestal_method_comparison")
    if not exists(fig_dir):
        print("Creating directory: {}".format(fig_dir))
        makedirs(fig_dir)

    pedestals = np.load(pedestal_path)
    ped_basic = pedestals['basic']
    ped_blocks4 = pedestals['blocks4']
    ped_blocks8 = pedestals['blocks8']
    ped_adrian = pedestals['adrian']
    ped_correction = pedestals['correction']

    # ped_basic[:, :, 4098:] = 0
    # ped_basic[:, :, :128] = 0
    # ped_basic[:, :, 3968:] = 0
    ped_basic = np.ma.masked_where(ped_basic == 0, ped_basic)
    # ped_blocks4[:, :, 4098:] = 0
    # ped_blocks4[:, :, :128] = 0
    # ped_blocks4[:, :, 3968:] = 0
    ped_blocks4 = np.ma.masked_where(ped_blocks4 == 0, ped_blocks4)
    # ped_blocks8[:, :, 4098:] = 0
    # ped_blocks8[:, :, :128] = 0
    # ped_blocks8[:, :, 3968:] = 0
    ped_blocks8 = np.ma.masked_where(ped_blocks8 == 0, ped_blocks8)
    # ped_adrian[:, :, 128:] = 0
    # ped_adrian[:, :, :3] = 0
    # ped_adrian[:, :, 125:] = 0
    ped_adrian = np.ma.masked_where(ped_adrian == 0, ped_adrian)
    # ped_correction[:, :, 4098:] = 0
    # ped_correction[:, :, :128] = 0
    # ped_correction[:, :, 3968:] = 0
    ped_correction = np.ma.masked_where(ped_correction == 0, ped_correction)

    reader = Reader(input_path)
    n_events = reader.n_events
    n_tmpix = reader.n_tmpix
    n_pix = reader.n_pix
    n_samples = reader.n_samples
    samples = reader.samples
    fci = reader.first_cell_ids

    tm = np.arange(n_pix, dtype=np.uint16) // n_tmpix
    tmpix = np.arange(n_pix, dtype=np.uint16) % n_tmpix

    pix_i = np.arange(n_pix)
    tm_i = tm[:, None] * np.ones((n_pix, n_samples), dtype=np.int)
    tmpix_i = tmpix[:, None] * np.ones((n_pix, n_samples), dtype=np.int)

    tm_ix = tm_i[..., None]
    tmpix_ix = tmpix_i[..., None]

    sample_range = np.arange(n_samples, dtype=np.int)[None, :]
    sample_zeros = np.zeros(n_samples, dtype=np.int)[None, :]
    sample_index = sample_range + np.zeros(n_pix, dtype=np.int)[:, None]

    max_events = 1000
    storage_basic = np.ma.zeros((max_events, n_pix, n_samples))
    storage_blocks4 = np.ma.zeros((max_events, n_pix, n_samples))
    storage_blocks8 = np.ma.zeros((max_events, n_pix, n_samples))
    storage_adrian = np.ma.zeros((max_events, n_pix, n_samples))
    storage_correction = np.ma.zeros((max_events, n_pix, n_samples))

    source = reader.event_generator()
    with tqdm(total=max_events) as pbar:
        for ev in source:
            if ev >= max_events:
                break

            pbar.update(1)

            uncorrected_cells = sample_range + fci[:, None]
            cells = uncorrected_cells % N_CELLS
            fci_cell = sample_zeros + fci[:, None]

            uncor_x = uncorrected_cells[:, :, None]
            cells_x = cells[:, :, None]
            fci_cell_x = fci_cell[:, :, None]

            i_block4 = uncor_x // N_BLOCKSAMPLES - \
                (uncor_x[:, 0, :] // N_BLOCKSAMPLES)[:, None]
            i_block8 = uncor_x // (N_BLOCKSAMPLES // 2) - \
                (uncor_x[:, 0, :] // (N_BLOCKSAMPLES // 2))[:, None]
            bp_fci, r_fci, c_fci = get_bp_r_c(fci_cell_x)
            block_fci = r_fci + c_fci * N_ROWS
            block_pos = sample_range[:, :, None] + bp_fci

            s = samples[pix_i]
            s_x = s[..., None]

            p = ped_basic[tm_i, tmpix_i, cells]
            pedsub_basic = s - p
            storage_basic[ev] = pedsub_basic

            p = ped_blocks4[tm_ix, tmpix_ix, cells_x, i_block4]
            pedsub_blocks4 = (s_x - p)[..., 0]
            storage_blocks4[ev] = pedsub_blocks4

            p = ped_blocks8[tm_ix, tmpix_ix, cells_x, i_block8]
            pedsub_blocks8 = (s_x - p)[..., 0]
            storage_blocks8[ev] = pedsub_blocks8

            p = ped_adrian[tm_ix, tmpix_ix, block_fci, block_pos]
            pedsub_adrian = (s_x - p)[..., 0]
            storage_adrian[ev] = pedsub_adrian

            p = ped_correction[tm_i, tmpix_i, sample_index]
            pedsub_correction = pedsub_basic - p
            storage_correction[ev] = pedsub_correction

    plt.style.use("ggplot")

    pixel = 0
    fig_pixel_avg = plt.figure(figsize=(6, 6))
    ax_pixel_avg = fig_pixel_avg.add_subplot(1, 1, 1)
    ax_pixel_avg.set_title('Pixel {} Average Residuals'.format(pixel))
    ax_pixel_avg.set_xlabel('Sample')

    event = 2
    fig_event_avg = plt.figure(figsize=(6, 6))
    ax_event_avg = fig_event_avg.add_subplot(1, 1, 1)
    ax_event_avg.set_title('Event {} Average Residuals'.format(event))
    ax_event_avg.set_xlabel('Sample')

    fig_all_hist = plt.figure(figsize=(10, 10))
    ax_all_hist = fig_all_hist.add_subplot(1, 1, 1)
    ax_all_hist.set_title('All Samples (N_events={})'.format(max_events))
    ax_all_hist.set_xlabel('ADC Residual')

    fig_blocks4_hist = plt.figure(figsize=(10, 10))
    ax_blocks4_hist = fig_blocks4_hist.add_subplot(1, 1, 1)
    ax_blocks4_hist.set_title('All Samples, All Events (Blocks4)')
    ax_blocks4_hist.set_xlabel('ADC Residual')

    n_plots = 9
    fig_wf_pixels = plt.figure(figsize=(13, 13))
    fig_wf_pixels.suptitle('First 9 Pixels, Event 0')
    ax_wf_pixels_list = []
    fig_wf_events = plt.figure(figsize=(13, 13))
    fig_wf_events.suptitle('First 9 Events, Pixel 0')
    ax_wf_events_list = []
    for iax in range(n_plots):
        x = np.floor(np.sqrt(n_plots))
        y = np.ceil(np.sqrt(n_plots))
        ax = fig_wf_pixels.add_subplot(x, y, iax + 1)
        ax_wf_pixels_list.append(ax)
        ax = fig_wf_events.add_subplot(x, y, iax + 1)
        ax_wf_events_list.append(ax)

    pixel_mean_basic = np.mean(storage_basic, axis=0)
    pixel_mean_blocks4 = np.mean(storage_blocks4, axis=0)
    pixel_mean_blocks8 = np.mean(storage_blocks8, axis=0)
    pixel_mean_adrian = np.mean(storage_adrian, axis=0)
    pixel_mean_correction = np.mean(storage_correction, axis=0)

    event_mean_basic = np.mean(storage_basic, axis=1)
    event_mean_blocks4 = np.mean(storage_blocks4, axis=1)
    event_mean_blocks8 = np.mean(storage_blocks8, axis=1)
    event_mean_adrian = np.mean(storage_adrian, axis=1)
    event_mean_correction = np.mean(storage_correction, axis=1)

    mean_basic = np.asscalar(np.mean(storage_basic))
    mean_blocks4 = np.asscalar(np.mean(storage_blocks4))
    mean_blocks8 = np.asscalar(np.mean(storage_blocks8))
    mean_adrian = np.asscalar(np.mean(storage_adrian))
    mean_correction = np.asscalar(np.mean(storage_correction))

    stddev_basic = np.asscalar(np.std(storage_basic))
    stddev_blocks4 = np.asscalar(np.std(storage_blocks4))
    stddev_blocks8 = np.asscalar(np.std(storage_blocks8))
    stddev_adrian = np.asscalar(np.std(storage_adrian))
    stddev_correction = np.asscalar(np.std(storage_correction))

    ax_pixel_avg.plot(pixel_mean_basic[pixel], label='basic')
    ax_pixel_avg.plot(pixel_mean_blocks4[pixel], label='blocks4')
    ax_pixel_avg.plot(pixel_mean_blocks8[pixel], label='blocks8')
    ax_pixel_avg.plot(pixel_mean_adrian[pixel], label='adrian')
    ax_pixel_avg.plot(pixel_mean_correction[pixel], label='correction')
    ax_pixel_avg.legend(loc=1)

    ax_event_avg.plot(event_mean_basic[event], label='basic')
    ax_event_avg.plot(event_mean_blocks4[event], label='blocks4')
    ax_event_avg.plot(event_mean_blocks8[event], label='blocks8')
    ax_event_avg.plot(event_mean_adrian[event], label='adrian')
    ax_event_avg.plot(event_mean_correction[event], label='correction')
    ax_event_avg.legend(loc=1)

    ax_all_hist.hist(storage_basic.compressed(), 100, alpha=0.4,
                     label="basic, Mean: {:.3}, Stddev: {:.3}"
                     .format(mean_basic, stddev_basic))
    ax_all_hist.hist(storage_blocks4.compressed(), 100, alpha=0.4,
                     label="blocks4, Mean: {:.3}, Stddev: {:.3}"
                     .format(mean_blocks4, stddev_blocks4))
    ax_all_hist.hist(storage_blocks8.compressed(), 100, alpha=0.4,
                     label="blocks8, Mean: {:.3}, Stddev: {:.3}"
                     .format(mean_blocks8, stddev_blocks8))
    ax_all_hist.hist(storage_adrian.compressed(), 100, alpha=0.4,
                     label="adrian, Mean: {:.3}, Stddev: {:.3}"
                     .format(mean_adrian, stddev_adrian))
    ax_all_hist.hist(storage_correction.compressed(), 100, alpha=0.4,
                     label="correction, Mean: {:.3}, Stddev: {:.3}"
                     .format(mean_correction, stddev_correction))
    ax_all_hist.legend(loc=1)

    for i in range(n_plots):
        ax = ax_wf_pixels_list[i]
        ax.plot(storage_blocks4[0, i])
        ax = ax_wf_events_list[i]
        ax.plot(storage_blocks4[i, 0])

    path_pixel_avg = join(fig_dir, "pixel_avg.pdf")
    path_event_avg = join(fig_dir, "event_avg.pdf")
    path_all_hist = join(fig_dir, "comparison_hist.pdf")
    path_all_hist_log = join(fig_dir, "comparison_hist_log.pdf")
    path_wf_pixels = join(fig_dir, "wf_pixels.pdf")
    path_wf_events = join(fig_dir, "wf_events.pdf")

    fig_pixel_avg.savefig(path_pixel_avg)
    print("Created figure: {}".format(path_pixel_avg))
    fig_event_avg.savefig(path_event_avg)
    print("Created figure: {}".format(path_event_avg))
    fig_all_hist.savefig(path_all_hist)
    print("Created figure: {}".format(path_all_hist))
    ax_all_hist.set_yscale('log')
    fig_all_hist.savefig(path_all_hist_log)
    print("Created figure: {}".format(path_all_hist_log))
    fig_wf_pixels.savefig(path_wf_pixels)
    print("Created figure: {}".format(path_wf_pixels))
    fig_wf_events.savefig(path_wf_events)
    print("Created figure: {}".format(path_wf_events))

    # Second loop with all events
    n_bins = 100
    edges = np.histogram(-1, bins=n_bins, range=[-50, 50])[1]
    between = (edges[1:] + edges[:-1]) / 2
    hist = np.zeros(n_bins)

    # Loop with all events
    source = reader.event_generator()
    with tqdm(total=n_events) as pbar:
        for _ in source:
            pbar.update(1)

            uncorrected_cells = sample_range + fci[:, None]
            cells = uncorrected_cells % N_CELLS

            uncor_x = uncorrected_cells[:, :, None]
            cells_x = cells[:, :, None]

            i_block4 = uncor_x // N_BLOCKSAMPLES - \
                (uncor_x[:, 0, :] // N_BLOCKSAMPLES)[:, None]

            s = samples[pix_i]
            s_x = s[..., None]

            p = ped_blocks4[tm_ix, tmpix_ix, cells_x, i_block4]
            pedsub_blocks4 = (s_x - p)[..., 0]
            hist += np.histogram(pedsub_blocks4.compressed(), edges)[0]

    average = np.average(between, weights=hist)
    variance = np.average((between - average) ** 2, weights=hist)
    stddev = np.sqrt(variance)

    ax_blocks4_hist.hist(between, edges, weights=hist, alpha=0.4,
                         label="blocks4, Hist Mean: {:.3}, Hist Stddev: {:.3}".
                         format(average, stddev))
    ax_blocks4_hist.legend(loc=1)

    path_blocks4_hist = join(fig_dir, "blocks4_hist.pdf")
    path_blocks4_hist_log = join(fig_dir, "blocks4_hist_log.pdf")
    fig_blocks4_hist.savefig(path_blocks4_hist)
    print("Created figure: {}".format(path_blocks4_hist))
    ax_blocks4_hist.set_yscale('log')
    fig_blocks4_hist.savefig(path_blocks4_hist_log)
    print("Created figure: {}".format(path_blocks4_hist_log))


def main():
    description = 'Compare different pedestal techniques'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-g', '--generate', dest='generate',
                        action='store_true', default=False,
                        help='generate the pedestals')
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO run file')
    parser.add_argument('-P', '--pedestal', dest='ped_path', action='store',
                        required=True, help='path for the .npz pedestal file')

    args = parser.parse_args()

    if args.generate:
        print("Generating pedestal")
        generate_pedestal(args.input_path, args.ped_path)
    else:
        print("Plotting residuals")
        apply_pedestal(args.input_path, args.ped_path)

    print("FINISHED")


if __name__ == '__main__':
    main()
