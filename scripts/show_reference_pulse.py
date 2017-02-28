import numpy as np
from matplotlib import pyplot as plt

from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.io import CameraGeometry


class ReferencePulseDisplayer(Component):
    name = 'ReferencePulseDisplayer'

    def __init__(self, config, tool, **kwargs):
        """
        Plotter for camera images.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

    @staticmethod
    def show_information(event, telid):
        geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                    event.inst.optical_foclen[telid])

        n_chan, n_pixels, n_samples = event.r0.tel[telid].adc_samples.shape
        waveforms = event.r0.tel[telid].adc_samples[0]
        max_charges = np.max(waveforms, axis=1)
        max_pix = int(np.argmax(max_charges))
        waveform = waveforms[max_pix]

        refshapes = event.mc.tel[telid].reference_pulse_shape
        refstep = event.mc.tel[telid].meta['refstep']
        time_slice = event.mc.tel[telid].time_slice

        x_wf = np.arange(n_samples) * time_slice
        x_ref = np.arange(refshapes.shape[1]) * refstep

        print("Event: {}".format(event.count))
        print("Telescope ID: {}".format(telid))
        print("Telescope type: {}".format(geom.cam_id))
        print("Num Channels: {}".format(n_chan))
        print("Num Pixels: {}".format(n_pixels))
        print("Num Samples: {}".format(n_samples))
        print("Sample Time: {} ns".format(time_slice))
        print("Num Reference Pulse Channels: {}".format(refshapes.shape[0]))
        print("Num Reference Pulse Samples: {}".format(refshapes.shape[1]))
        print("Reference Pulse Samples Time: {} ns".format(refstep))
        print(refshapes[0])
        print("")

        # Draw figures
        fig = plt.figure(figsize=(18, 9))
        fig.subplots_adjust(hspace=.5)

        ax_wf = fig.add_subplot(2, 1, 1)
        ax_ref = fig.add_subplot(2, 1, 2)

        ax_wf.plot(x_wf, waveform)
        ax_ref.plot(x_ref, refshapes[0])

        plt.show()


class ReferencePulseInvestigator(Tool):
    name = "ReferencePulseInvestigator"
    description = "Show information about the reference pulse in a file"

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ))
    classes = List([EventFileReaderFactory,
                    ReferencePulseDisplayer,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.displayer = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        self.displayer = ReferencePulseDisplayer(**kwargs)

    def start(self):
        source = self.file_reader.read()
        for event in source:
            for telid in list(event.r0.tels_with_data):
                self.displayer.show_information(event, telid)

    def finish(self):
        pass


if __name__ == '__main__':
    exe = ReferencePulseInvestigator()
    exe.run()
