import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import layout
from bokeh.models import HoverTool, ColumnDataSource
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from tqdm import tqdm
import argparse
from os import makedirs
from os.path import splitext, join, dirname, basename, exists

# CHEC-M
# N_ROWS = 8
# N_COLUMNS = 64
# N_BLOCKS = N_ROWS * N_COLUMNS
# N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
# SKIP_SAMPLE = 32
# SKIP_END_SAMPLE = 0
# SKIP_EVENT = 2
# SKIP_END_EVENT = 1

# CHEC-S
N_ROWS = 8
N_COLUMNS = 16
N_BLOCKS = N_ROWS * N_COLUMNS
N_CELLS = N_ROWS * N_COLUMNS * N_BLOCKSAMPLES
SKIP_SAMPLE = 0
SKIP_END_SAMPLE = 0
SKIP_EVENT = 2
SKIP_END_EVENT = 1


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


class AdcSpread:
    def __init__(self):
        self.layout = None

    def create(self, mean, stddev, min_, max_):
        x = np.arange(mean.size)

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "ADC Spread Vs Event"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("(min, max)", "@min, @max")
                                          ], renderers=[c]))

        fig.xaxis.axis_label = 'Event'
        fig.yaxis.axis_label = 'ADC'

        # Rangebars
        # top = max_
        # bottom = min_
        # left = x - 0.3
        # right = x + 0.3
        # fig.segment(x0=x, y0=bottom, x1=x, y1=top,
        #             line_width=1.5, color='red')
        # fig.segment(x0=left, y0=top, x1=right, y1=top,
        #             line_width=1.5, color='red')
        # fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
        #             line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


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

    shape = (n_modules, n_tmpix, N_BLOCKS, max_blocksinwf * N_BLOCKSAMPLES)
    ped = np.zeros(shape)
    hits = np.zeros(shape)

    pix_i = np.arange(n_pix)
    tm_i = tm[:, None] * np.ones((n_pix, n_samples), dtype=np.int)
    tmpix_i = tmpix[:, None] * np.ones((n_pix, n_samples), dtype=np.int)

    tm_ix = tm_i[..., None]
    tmpix_ix = tmpix_i[..., None]

    sample_range = np.arange(n_samples, dtype=np.int)[None, :]
    sample_zeros = np.zeros(n_samples, dtype=np.int)[None, :]

    def mean(pedestal, n, samp):
        return pedestal * (n / (n + 1)) + samp / (n + 1)

    source = reader.event_generator()
    n_events = reader.n_events
    with tqdm(total=n_events) as pbar:
        for _ in source:
            pbar.update(1)

            fci_cell = sample_zeros + fci[:, None]
            fci_cell_x = fci_cell[:, :, None]
            bp_fci, r_fci, c_fci = get_bp_r_c(fci_cell_x)
            block_fci = r_fci + c_fci * N_ROWS
            block_pos = sample_range[:, :, None] + bp_fci

            s = samples[pix_i]
            s_x = s[..., None]

            p = ped[tm_ix, tmpix_ix, block_fci, block_pos]
            h = hits[tm_ix, tmpix_ix, block_fci, block_pos]
            ped[tm_ix, tmpix_ix, block_fci, block_pos] = mean(p, h, s_x)
            hits[tm_ix, tmpix_ix, block_fci, block_pos] += 1

    if not exists(dirname(output_path)):
        print("Creating directory: {}".format(dirname(output_path)))
        makedirs(dirname(output_path))
    np.save(output_path, ped)
    print("Pedestal saved to: {}".format(output_path))


def apply_pedestal(input_path, pedestal_path):
    ped = np.load(pedestal_path)
    ped = np.ma.masked_where(ped == 0, ped)

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

    event_mean = np.zeros(n_events)
    event_stddev = np.zeros(n_events)
    event_min = np.zeros(n_events)
    event_max = np.zeros(n_events)

    source = reader.event_generator()
    for ev in tqdm(source, total=n_events):
        fci_cell = sample_zeros + fci[:, None]
        fci_cell_x = fci_cell[:, :, None]
        bp_fci, r_fci, c_fci = get_bp_r_c(fci_cell_x)
        block_fci = r_fci + c_fci * N_ROWS
        block_pos = sample_range[:, :, None] + bp_fci

        s = samples[pix_i]
        s_x = s[..., None]

        p = ped[tm_ix, tmpix_ix, block_fci, block_pos]
        pedsub = (s_x - p)[..., 0]

        event_mean[ev] = np.mean(pedsub)
        event_stddev[ev] = np.std(pedsub)
        event_min[ev] = np.min(pedsub)
        event_max[ev] = np.max(pedsub)

    # Create bokeh figures
    p_adcspread = AdcSpread()
    p_adcspread.create(event_mean, event_stddev, event_min, event_max)

    # Get bokeh layouts
    l_adcspread = p_adcspread.layout

    # Layout
    layout_list = [
        [l_adcspread]
    ]
    l = layout(layout_list, sizing_mode="scale_width")

    output_dir = join(dirname(input_path), splitext(basename(input_path))[0],
                      "simple_pedestal_residuals")
    if not exists(output_dir):
        print("Creating directory: {}".format(output_dir))
        makedirs(output_dir)

    path = join(output_dir, 'adc_drift.html')
    output_file(path)
    show(l)
    print("Created bokeh figure: {}".format(path))


def main():
    description = 'Compare different pedestal techniques'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-g', '--generate', dest='generate',
                        action='store_true', default=False,
                        help='generate the pedestal file')
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO run file')
    parser.add_argument('--ped', '--pedestal', dest='ped_path', action='store',
                        required=True, help='path for the .npy pedestal file')

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
