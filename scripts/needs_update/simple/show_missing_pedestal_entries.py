import numpy as np
from bokeh.models import HoverTool
from bokeh.plotting import figure, output_file, show
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


def show_absent_pedestal_entries(input_path, pedestal_path):

    fig_dir = join(dirname(input_path), splitext(basename(input_path))[0])
    if not exists(fig_dir):
        print("Creating directory: {}".format(fig_dir))
        makedirs(fig_dir)

    pedestals = np.load(pedestal_path)
    ped_blocks4 = pedestals['blocks4']
    ped_adrian = pedestals['adrian']

    tm = 0
    tmpix = 0

    p = ped_adrian[tm, tmpix]
    # p = ped_blocks4[tm, tmpix]

    empty_x, empty_y = np.where(p == 0)

    hits = np.zeros(p.shape)

    reader = Reader(input_path)
    n_samples = reader.n_samples
    sample_range = np.arange(n_samples, dtype=np.int)
    sample_zeros = np.zeros(n_samples, dtype=np.int)[None, :]
    source = reader.event_generator()
    with tqdm(total=reader.n_events) as pbar:
        for _ in source:
            pbar.update(1)
            fci = reader.first_cell_ids[0]
            bp, r, c = get_bp_r_c(fci)
            block = r + c * N_ROWS
            hits[block, bp:bp+n_samples] += 1

            # fci = reader.first_cell_ids[0]
            # uncorrected_cells = sample_range + fci
            # cells = uncorrected_cells % N_CELLS
            # uncor_x = uncorrected_cells[:, None]
            # cells_x = cells[:, None]
            # i_block4 = uncor_x // N_BLOCKSAMPLES - \
            #     (uncor_x[0, :] // N_BLOCKSAMPLES)[:, None]
            # h = hits[cells_x, i_block4]
            # hits[cells_x, i_block4] = h + 1

    hit_x, hit_y = np.where((p == 0) & (hits > 0))

    fig_dir = join(dirname(input_path), splitext(basename(input_path))[0])
    if not exists(fig_dir):
        print("Creating directory: {}".format(fig_dir))
        makedirs(fig_dir)

    output_file(join(fig_dir, 'absent.html'))
    fig = figure(plot_width=700, plot_height=700,
                 active_scroll='wheel_zoom')
    fig.xaxis.axis_label="Block"
    fig.yaxis.axis_label="BlockPhase+Sample"
    # fig.xaxis.axis_label = "Cell"
    # fig.yaxis.axis_label = "DBlock"
    c1 = fig.circle(empty_x, empty_y, legend="Pedestal_Hits=0")
    c2 = fig.cross(hit_x, hit_y, size=10, color='orange',
                   legend="Pedestal_Hits=0, Run_Hits>0")
    fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                            renderers=[c1, c2]))
    show(fig)


def main():
    description = 'Check for missing pedestal values'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO run file')
    parser.add_argument('-P', '--pedestal', dest='ped_path', action='store',
                        required=True, help='path for the .npz pedestal file')

    args = parser.parse_args()

    show_absent_pedestal_entries(args.input_path, args.ped_path)

    print("FINISHED")


if __name__ == '__main__':
    main()
