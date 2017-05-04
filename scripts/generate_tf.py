"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List, Int
from ctapipe.core import Tool
from targetpipe.io.file_looper import TargetioFileLooper
from targetpipe.calib.camera.makers import TFMaker
from tqdm import tqdm


class TFBuilder(Tool):
    name = "TFBuilder"
    description = "Create the TargetCalib Transfer Function file from a " \
                  "list of event files"

    vped_list = List(Int, None, allow_none=True,
                     help='List of the vped value for each input '
                          'file').tag(config=True)

    aliases = Dict(dict(N='TargetioFileLooper.max_files',
                        max_events='TargetioFileLooper.max_events',
                        P='TFMaker.pedestal_path',
                        adcstep='TFMaker.adc_step',
                        O='TFMaker.output_path',

                        ))
    flags = Dict(dict(compress=({'TFMaker': {'compress': True}},
                                'Compress the output tf file (store '
                                'in uint16 instead of floats'),
                      input=({'TFMaker': {'tf_input': True}},
                             'Create a numpy file containing the input TF'
                             'array before the switch of axis')
                      ))
    classes = List([TargetioFileLooper,
                    TFMaker
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_looper = None
        self.tfmaker = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_looper = TargetioFileLooper(**kwargs)

        first_event = self.file_looper.file_reader_list[0].get_event(0)
        n_modules = first_event.meta['n_modules']

        self.tfmaker = TFMaker(**kwargs,
                               vped_list=self.vped_list,
                               number_tms=n_modules)

        assert len(self.file_looper.file_reader_list) == len(self.vped_list)

    def start(self):
        desc1 = "Looping over Files"
        desc2 = "Looping over events"
        iterable = zip(self.file_looper.file_reader_list, self.vped_list)
        n_readers = self.file_looper.num_readers
        for reader, vped in tqdm(iterable, total=n_readers, desc=desc1):
            n_events = reader.num_events
            source = reader.read()
            for event in tqdm(source, total=n_events, desc=desc2):
                self.tfmaker.add_event(event, vped)

    def finish(self):
        self.tfmaker.save()


exe = TFBuilder()
exe.run()
