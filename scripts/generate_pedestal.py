"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.calib.camera.makers import PedestalMaker
from tqdm import tqdm


class PedestalBuilder(Tool):
    name = "PedestalBuilder"
    description = "Create the TargetCalib Pedestal file from waveforms"

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        O='PedestalMaker.output_path',
                        ))
    flags = Dict(dict(compress=({'PedestalMaker': {'compress': True}},
                                'Compress the output pedestal file (store '
                                'in uint16 instead of floats'),
                      stddev=({'PedestalMaker': {'stddev': True}},
                              'Create a numpy file containing the standard '
                              'deviation of the pedestal')
                      ))
    classes = List([EventFileReaderFactory,
                    PedestalMaker
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.pedmaker = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        first_event = self.file_reader.get_event(0)
        n_modules = first_event.meta['n_modules']
        n_blocks = first_event.meta['n_blocks']
        n_samples = first_event.r0.tel[0].adc_samples.shape[2]

        self.pedmaker = PedestalMaker(**kwargs,
                                      n_tms=n_modules,
                                      n_blocks=n_blocks,
                                      n_samples=n_samples)

    def start(self):
        n_events = self.file_reader.num_events
        desc = "Filling pedestal"
        with tqdm(total=n_events, desc=desc) as pbar:
            source = self.file_reader.read()
            for event in source:
                pbar.update(1)
                self.pedmaker.add_event(event)

    def finish(self):
        self.pedmaker.save()


exe = PedestalBuilder()
exe.run()
