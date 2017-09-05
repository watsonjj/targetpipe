import argparse
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from os.path import join, dirname, basename, splitext
from target_io import TargetIOEventReader as TIOReader
from target_io import T_SAMPLES_PER_WAVEFORM_BLOCK as N_BLOCKSAMPLES
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Colormap
import seaborn as sns

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

        self.is_r1 = self.reader.fR1
        if not self.is_r1:
            raise IOError("This script is only setup to read *_r1.tio files!")

        self.n_events = self.reader.fNEvents
        self.run_id = self.reader.fRunID
        self.n_pix = self.reader.fNPixels
        self.n_modules = self.reader.fNModules
        self.n_tmpix = self.n_pix // self.n_modules
        self.n_samples = self.reader.fNSamples
        self.n_cells = self.reader.fNCells

        self.max_blocksinwf = self.n_samples // N_BLOCKSAMPLES + 1
        self.samples = np.zeros((self.n_pix, self.n_samples), dtype=np.float32)
        self.first_cell_ids = np.zeros(self.n_pix, dtype=np.uint16)

        directory = dirname(path)
        filename = splitext(basename(path))[0]
        self.plot_directory = join(directory, filename)

    def get_event(self, iev):
        self.reader.GetR1Event(iev, self.samples, self.first_cell_ids)

    def event_generator(self):
        for iev in range(self.n_events):
            self.get_event(iev)
            yield iev


def main():
    description = 'Check for missing pedestal values'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--file', dest='input_path', action='store',
                        required=True, help='path to the TIO r1 run file')
    args = parser.parse_args()

    reader = Reader(args.input_path)
    source = reader.event_generator()

    hist_max = np.zeros(shape=(reader.n_pix,reader.n_events))
    #hist_max = np.zeros(shape=(reader.n_pix,400))

    for ev in source:
        # Skip first row due to problem in pedestal subtraction
        bp, r, c = get_bp_r_c(reader.first_cell_ids[0])
        if r == 0:
            continue
        print(ev)

        waveforms = reader.samples

        waveforms_diff = np.zeros(waveforms.shape)

        for i in range(1,waveforms.shape[1]):
            waveforms_diff[:,i] = waveforms[:,i] - waveforms[:,i-1]
        for i in range(waveforms.shape[0]):
            if (ev == 3 and i == 35):
                plt.plot(waveforms[i], label='Waveform')
                plt.plot(waveforms_diff[i], label='Differentiated waveform')
                plt.xlabel('Time (ns)')
                plt.ylabel('Pedestal subtracted signal (ADC)')
                plt.legend()
                plt.show()
            if (waveforms[i].max() > 100):# and ev < 400):
                # differentiate waveform and look for the maximum, then take a weighted mean around that maximum to
                # get a better estimate of the real maximum of the differentiated curve
                w = waveforms_diff[i] / waveforms_diff[i].max()
                c = np.where(w==w.max())[0][0]
                center = np.arange(c-3,c+4,1)
                max_x = np.average(center,weights=w[c-3:c+4])
                hist_max[i,ev]=max_x
                if ev == 4 and i == 35:
                    plt.plot(waveforms[i], label='Waveform')
                    plt.plot(waveforms_diff[i],label = 'Differentiated waveform')
                    plt.xlabel('Time (ns)')
                    plt.ylabel('Pedestal subtracted signal (ADC)')
                    plt.legend()
                    plt.show()
            else:
                pass
                #plt.plot(waveforms[i],'r')
                #plt.title(str(i))
                #plt.show()

    return hist_max


if __name__ == '__main__':
    hist_max = main()
    for e in range(1):#len(hist_max)):
        hist_ma =  ma.masked_where(hist_max[:,e] == 0, hist_max[:,e])
        first = ma.masked_where(hist_max[:,e] == 0, hist_max[:,e]).min()
        last = np.max(hist_max[:,e])
        plt.hist(hist_max[:,e],bins=10,range=(first,last))
        #labels, counts = np.unique(hist_max[e], return_counts=True)
        #plt.bar(labels, counts, align='center',alpha=0.5)
        #plt.gca().set_xticks(np.linspace(first,last,(last-first)+1))
        plt.xlabel("Rising edge (ns)")
        plt.text(31.5,5,r"stddev=%.2f" % (hist_ma.std()))
        plt.show()

    p=32
    plt.hist(hist_max[p],range=(29,35),bins=14)
    plt.xlabel('Rising edge (ns)')
    plt.yscale('log')
    plt.text(34,1000,'stddev = %.2f' % ma.masked_where(hist_max[p]==0,hist_max[p]).std())
    plt.show()

    # 3D histogram
    # x_data, y_data = np.meshgrid( np.arange(hist_max.shape[1]), np.arange(hist_max.shape[0]) )
    #
    # x_data = x_data.flatten()
    # y_data = y_data.flatten()
    # z_data = hist_max.flatten()
    #
    # # with colorbar
    # cmap = plt.cm.get_cmap('RdYlBu')
    # # cmap=plt.cm.jet
    # Colormap.set_under(cmap,color='k')
    #
    # cs = plt.scatter(x_data,y_data,c=z_data,cmap=cmap)#,vmin=30)
    # plt.xlabel('Event')
    # plt.ylabel('Pixel')
    # plt.title('Rise Time (ns)')
    # plt.colorbar(cs)
    # plt.show()



    # with third dimension
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.bar3d( x_data,y_data,np.zeros(len(z_data)),1, 1, z_data)
    # ax.set_ylabel('Pixel')
    # ax.set_xlabel('Event')
    # ax.set_zlabel('Rising edge time (ns)')
    # plt.show()


    std_m = []
    for e in range(len(hist_max)):
        try:
            sns.kdeplot(hist_max.T[e][32:63],kernel='gau')
            std_m.append(hist_max.T[e][32:63].std())
        except:
            pass
    plt.xlabel('Rising edge (ns)')
    plt.ylabel('Density')
    plt.text(34, 0.6, 'mean stddev = %.2f' % np.mean(std_m))
    plt.show()

    pixel = np.arange(32,63)
    std_m = []
    for p in pixel:
        c = ma.masked_where(hist_max[p]==0,hist_max[p])
        sns.kdeplot(hist_max[p],kernel='gau',legend=True)
        std_m.append(c.std())
    plt.text(34, 0.18, 'mean stddev = %.2f' % np.mean(std_m))
    plt.xlabel('Rising edge (ns)')
    plt.ylabel('Density')
    plt.xlim([22,43])
    plt.show()



