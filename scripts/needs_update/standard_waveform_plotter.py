from traitlets import Dict, List, Bool
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.chec import CHECMSPEFitter
from targetpipe.io.pixels import Dead
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import  Rectangle
from os.path import realpath, join, dirname
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from os import makedirs


class DL1Extractor(Tool):
    name = "DL1Extractor"
    description = "Extract the dl1 information and store into a numpy file"

    plot_all = Bool(False, help='Whether to make a directory and plot'
                                   'all events in the run. Better for small'
                                   'runs').tag(config=True)

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        plot='DL1Extractor.plot_all'
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMSPEFitter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.event_no = None
        self.cleaner = None
        self.extractor = None
        self.fitter = None
        self.dead = None
        self.t0 = None
        self.peak_height = None
        self.output_dir = None
        self.times = None
        self.adc2pe = None
        self.charge = None
        self.rms = None
        self.fig = None
        self.rms_sum = None

    def _plot_camera(self, im, ax, clip=False, temp_mask=None):
        path = join(realpath(dirname(__file__)), "../targetpipe/io/checm_pixel_pos.npy")
        pixel_pos = np.load(path)
        if temp_mask != None:
            mask = [temp_mask>10]
        width = 0.005  # abs(pixel_pos[0,0]-pixel_pos[0,1]*0.45)
        height = width  # abs(pixel_pos[1,0]-pixel_pos[1,2]*0.45)
        if clip:
            upper = 53
            lower = 33
            im[im>upper]=upper
            im[im<lower]=lower

        patches = []
        for i in range(len(pixel_pos[0])):
            if clip:
                if im[i] > upper or im[i] < lower:
                    pass
                else:
                    rect = Rectangle((pixel_pos[:, i] - [width / 2, height / 2]), width, height)
                    patches.append(rect)
            else:
                rect = Rectangle((pixel_pos[:, i] - [width / 2, height / 2]), width, height)
                patches.append(rect)
        if temp_mask != None:
            patches = [patches[i] for i in range(len(patches)) if mask[0][i]==True]
            im = im[mask]

        patches = PatchCollection(patches)
        patches.set_array(np.array(im))

        ax.add_collection(patches)
        self.fig.colorbar(patches, ax=ax)

        ax.set_xlim(min(pixel_pos[0]) - width, max(pixel_pos[0]) + width)
        ax.set_ylim(min(pixel_pos[1]) - height, max(pixel_pos[1]) + height)
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        return self.fig

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.cleaner = CHECMWaveformCleaner(**kwargs)
        self.extractor = CHECMExtractor(**kwargs)
        self.fitter = CHECMSPEFitter(**kwargs)
        self.dead = Dead()

        self.output_dir = join(self.reader.output_directory, "extract_adc2pe")
        if not exists(self.output_dir):
            self.log.info("Creating directory: {}".format(self.output_dir))
            makedirs(self.output_dir)

        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        n_pixels, n_samples = first_event.r0.tel[0].adc_samples[0].shape

        self.charge = np.zeros((n_events, n_pixels))

    def start(self):
        n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        n_pixels, n_samples = first_event.r0.tel[telid].adc_samples[0].shape
        self.times = np.zeros(n_events)

        # Prepare storage array
        area = np.zeros((n_events, n_pixels))
        ratio = np.ma.zeros(n_pixels)
        ratio.mask = np.zeros(ratio.shape, dtype=np.bool)
        self.rms = np.zeros(n_pixels)
        ratio.fill_value = 0

        source = self.reader.read()
        desc = "Looping through file"


        self.fig, ax = plt.subplots(4,2, figsize=(10,20))
        ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = ax
        ax1.set_title('Individual Pixel Waveforms')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('ADC')

        ax2.set_title('Target Module Average Waveforms')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Mean ADC')

        ax3.set_title('Histogram of Extracted Pixel Charge by Event')
        ax3.set_xlabel('Event')
        ax3.set_ylabel('Extracted Charge')

        ax4.set_title('Histogram of Peak Time by Event')
        ax4.set_xlabel('Event')
        ax4.set_ylabel('Peak Time [sample #]')

        ax5.set_title('Extracted Charge by Pixel')
        ax5.set_aspect('equal')

        ax6.set_title('Peak Height by Pixel')
        ax6.set_aspect('equal')

        ax7.set_title('Waveform RMS by Pixel')
        ax7.set_aspect('equal')

        ax8.set_title('Peak Time by Pixel')
        ax8.set_aspect('equal')

        self.rms_sum = np.zeros(n_pixels)

        self.peak_times = np.zeros((n_events, n_pixels))

        with tqdm(total=n_events, desc=desc) as pbar:
            for event_id, event in enumerate(source):
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

                # Perform CHECM Waveform Cleaning
                sb_sub_wf, t0 = self.cleaner.apply(dl0)

                # Perform CHECM Charge Extraction
                peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)
                self.rms = np.sqrt(np.mean((sb_sub_wf**2).T[10:86].T, axis=1))
                self.rms_sum += self.rms
                if self.plot_all: #20
                    # Hack: Shoud become a function to make this far less ugly in the loop
                    figz, bx = plt.subplots(3,2, figsize=(10,15))
                    ((bx1, bx2), (bx5, bx6), (bx7, bx8)) = bx
                    bx1.set_title('Individual Pixel Waveforms')
                    bx1.set_xlabel('Sample')
                    bx1.set_ylabel('ADC')

                    bx2.set_title('Target Module Average Waveforms')
                    bx2.set_xlabel('Sample')
                    bx2.set_ylabel('Mean ADC')

                    bx5.set_title('Extracted Charge by Pixel')
                    bx5.set_aspect('equal')

                    bx6.set_title('Peak Height by Pixel')
                    bx6.set_aspect('equal')

                    bx7.set_title('Waveform RMS by Pixel')
                    bx7.set_aspect('equal')

                    bx8.set_title('Peak Time by Pixel')
                    bx8.set_aspect('equal')

                    self.pix_number = event.count
                    [bx1.plot(np.arange(96), sb_sub_wf[i]) for i in range(1200,1208)]
                    [bx2.plot(np.arange(96), np.mean(sb_sub_wf[64*i:64*i+64],
                                                     axis=0), label='TM ' + str(i)) for i in range(32)]
                    # ax2.legend(ncol=3)
                    self.event_9 = sb_sub_wf
                    self.peak_height = np.array([np.max(i) for i in sb_sub_wf])  #peak_height
                    self.rms = np.sqrt(np.mean(sb_sub_wf**2, axis=1))

                    self.peak_times[index] += [np.argmax(i[10:-20])+10 for i in sb_sub_wf]
                    self.charge[index] = peak_area
                    self.times[index] = event.meta['tack']
                # self.rms += np.std(sb_sub_wf, axis=1)
                # self.rms += np.sqrt(np.mean(sb_sub_wf, axis=1)**2)
                    mask = self.charge.flatten() > -1000
                    self._plot_camera(self.charge[self.pix_number], bx5)
                    self._plot_camera(self.peak_height, bx6)
                    self._plot_camera(self.rms, bx7)
                    self._plot_camera(np.array([np.argmax(i) for i in self.event_9]), bx8, clip=True,
                                      temp_mask=np.array([np.max(i) for i in self.event_9]))
                    output_path = self.reader.input_path.replace(".tio", "/")\
                                  + self.reader.input_path.replace("_r0.tio", "_event_{}.pdf".format(event.count))
                    figz.savefig(output_path)
                    figz.close()

                if event.count == 9: #20
                    self.peak_times = np.zeros((n_events, n_pixels))

                    self.pix_number = event.count
                    [ax1.plot(np.arange(96), sb_sub_wf[i]) for i in range(1200,1208)]
                    [ax2.plot(np.arange(96), np.mean(sb_sub_wf[64*i:64*i+64], axis=0), label='TM '+str(i)) for i in range(32)]
                    # ax2.legend(ncol=3)
                    self.event_9 = sb_sub_wf
                    self.peak_height = np.array([np.max(i) for i in sb_sub_wf])  #peak_height
                    self.rms = np.sqrt(np.mean(sb_sub_wf**2, axis=1))

                self.peak_times[index] += [np.argmax(i[10:-20])+10 for i in sb_sub_wf]
                self.charge[index] = peak_area
                self.times[index] = event.meta['tack']
                # self.rms += np.std(sb_sub_wf, axis=1)
                # self.rms += np.sqrt(np.mean(sb_sub_wf, axis=1)**2)
        mask = self.charge.flatten() > -1000
        self._plot_camera(self.charge[self.pix_number], ax5)
        self._plot_camera(self.peak_height, ax6)
        self._plot_camera(self.rms, ax7)
        self._plot_camera(np.array([np.argmax(i) for i in self.event_9]), ax8, clip=True,
                          temp_mask=np.array([np.max(i) for i in self.event_9]))

        ax4.hist2d(np.array([np.ones(32*64)*i for i in np.arange(n_events)]).flatten(), self.peak_times.flatten(),
                   bins=((n_events, 66)))
        print(np.max(self.peak_times))
        ax3.hist2d(np.array([np.ones(32*64)*i for i in np.arange(n_events)]).flatten()[mask],
                   self.charge.flatten()[mask], bins=((n_events, 100)))

        #plt.tight_layout()

    def finish(self):
        output_path = self.reader.input_path.replace("_r0.tio", "_dl1.npz")
        output_path = output_path.replace("_r1.tio", "_dl1.tio")
        np.savez(output_path,
                 charge=self.charge
                 )
        # plt.tight_layout()
        self.fig.savefig(output_path.replace("_dl1.npz", "_waveform_plots.pdf"))
        self.log.info("Standard plots saved to: {}".format(output_path.replace("_dl1.npz", "_waveform_plots.pdf")))
        fig, ax = plt.subplots()
        self._plot_camera(self.rms_sum/self.reader.num_events, ax=ax) # - np.load('Run05273_summed_rms.npy')/8242
        np.save(output_path.replace("_dl1.npz", "_summed_rms.npy"), self.rms_sum/self.reader.num_events)
        fig.savefig(output_path.replace("_dl1.npz", "_summed_rms.pdf"))
        self.log.info("DL1 Numpy array saved to: {}".format(output_path))



exe = DL1Extractor()
exe.run()