import numpy as np
import pandas as pd
from bokeh.plotting import curdoc
from tqdm import tqdm
from traitlets import Dict, List, Unicode

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from targetpipe.io.file_looper import TargetioFileLooper as FileLooper
from targetpipe.plots.explore import ExploreDataApp

TELID = 0
ORIGIN = 'targetio'


class DataExplorer(Tool):
    name = "PedestalBuilder"
    description = "Create the TargetCalib Pedestal file from waveforms"

    test = Unicode('test').tag(config=True)

    aliases = Dict(dict(f='FileLooper.single_file',
                        N='FileLooper.max_files',
                        max_events='FileLooper.max_events',
                        query='ExploreDataApp.startup_query'
                        ))
    classes = List([FileLooper,
                    ExploreDataApp
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.extractor = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None

        self.file_looper = None
        self.app = None
        self.dataframe = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_looper = FileLooper(**kwargs)
        self.app = ExploreDataApp(**kwargs)

        ext = 'LocalPeakIntegrator'
        extractor_factory = ChargeExtractorFactory(**kwargs, extractor=ext)
        extractor_class = extractor_factory.get_class()
        self.extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=ORIGIN, **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor, **kwargs)

    def start(self):
        n_events = self.file_looper.num_events

        l = []

        desc = "Filling DataFrame"
        with tqdm(total=n_events, desc=desc) as pbar:
            for fn, fr in enumerate(self.get_next_file()):

                first_event = fr.get_event(0)

                # Find which TMs are connected
                first_waveforms = first_event.r0.tel[TELID].adc_samples[0]
                haspixdata = (first_waveforms.sum(axis=1) != 0)
                n_haspixdata = sum(haspixdata)
                self.log.info("Number of pixels with data = {}"
                              .format(n_haspixdata))

                source = fr.read()
                for ev, event in enumerate(source):
                    pbar.update(1)
                    d = self.get_event_dict(event, haspixdata)
                    d['file'] = fr.input_path
                    d['file_num'] = fn
                    d['tm'] = event.meta['tm']
                    d['tmpix'] = event.meta['tmpix']
                    l.append(pd.DataFrame(d))
        self.dataframe = pd.concat(l)

    def finish(self):
        self.app.start(self.dataframe)
        curdoc().add_root(self.app.layout)
        curdoc().title = self.app.name

    def get_next_file(self):
        kwargs = dict(config=self.config, tool=self)
        for fr in self.file_looper.file_reader_list:
            self.dl1 = CameraDL1Calibrator(extractor=self.extractor, **kwargs)
            yield fr

    def get_event_dict(self, event, pix_mask):
        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.dl1.calibrate(event)

        samples = event.r0.tel[TELID].adc_samples[0]
        n_pixels, n_samples = samples.shape

        charge = event.dl1.tel[TELID].image[0]
        first_sample = samples[:, 0]
        fci = event.r0.tel[TELID].first_cell_ids

        d = dict(event=event.count,
                 pixel=np.arange(n_pixels),
                 charge=charge,
                 first_sample=first_sample,
                 fci=fci)

        return d

# if __name__ == '__main__':
exe = DataExplorer()
exe.run()
