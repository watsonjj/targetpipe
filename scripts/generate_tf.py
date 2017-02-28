"""
Create a pedestal file from an event file using the target_calib Pedestal
class
"""

from traitlets import Dict, List, Int
from ctapipe.core import Tool
from targetpipe.io.eventfilereader import FileLooper
from targetpipe.calib.camera.makers import TFMaker
from tqdm import tqdm


class TFEventFileLooper(FileLooper):
    name = 'TFEventFileLooper'

    vped_list = List(Int, None, allow_none=True,
                     help='List of the vped value for each input '
                          'file').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)
        assert (len(self.file_list) == len(self.vped_list))

    def read(self):
        for fn, (filepath, vped) in enumerate(zip(self.file_list,
                                                  self.vped_list)):

            if self.max_files is not None:
                if fn >= self.max_files:
                    break

            self.file_reader.input_path = filepath
            self.file_reader.max_events = self.max_events

            source = self.file_reader.read()
            for event in source:
                yield fn, event, vped


class TFBuilder(Tool):
    name = "TFBuilder"
    description = "Create the TargetCalib Transfer Function file from a " \
                  "list of event files"

    aliases = Dict(dict(N='TFEventFileLooper.max_files',
                        max_events='TFEventFileLooper.max_events',
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
    classes = List([TFEventFileLooper,
                    TFMaker
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_looper = None
        self.tfmaker = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        self.file_looper = TFEventFileLooper(**kwargs)

        _, first_event, _ = next(self.file_looper.read())
        n_modules = first_event.meta['n_modules']

        self.tfmaker = TFMaker(**kwargs,
                               vped_list=self.file_looper.vped_list,
                               number_tms=n_modules)

    def start(self):
        n_events = self.file_looper.num_events
        desc = "Filling TF"
        with tqdm(total=n_events, desc=desc) as pbar:
            source = self.file_looper.read()
            for fn, event, vped in source:
                pbar.update(1)
                self.tfmaker.add_event(event, vped)

    def finish(self):
        self.tfmaker.save()


exe = TFBuilder()
exe.run()
