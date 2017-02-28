from bokeh.io import curdoc
from bokeh.layouts import widgetbox, layout
from traitlets import Dict, List
from bokeh.models import Select, TextInput, PreText
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.charge_extractors import ChargeExtractorFactory
from targetpipe.plots.event_viewer import EventViewer


class BokehFileViewer(Tool):
    name = "BokehFileViewer"
    description = "Interactively explore an event file"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        extractor='ChargeExtractorFactory.extractor',
                        window_width='ChargeExtractorFactory.window_width',
                        window_start='ChargeExtractorFactory.window_start',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        ))
    flags = Dict(dict(id=({'DisplayDL1Calib': {'use_event_index': True}},
                          'event_index will obtain an event using '
                          'event_id instead of index.')
                      ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraR1CalibratorFactory,
                    CameraDL1Calibrator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = None
        self._event_index = None
        self._event_id = None
        self._telid = None
        self._channel = None
        self._extractor = None

        self.w_event_index = None
        self.w_event_id = None
        self.w_telid = None
        self.w_channel = None
        self.w_dl1_dict = None
        self.wb_extractor = None
        self.layout = None

        self.file_reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.viewer = None

        self._updating_extractor = False

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        self._extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor, **kwargs)

        self.viewer = EventViewer(**kwargs)

        # Setup widgets
        self.viewer.create()
        self.viewer.enable_automatic_index_increment()
        self.create_event_index_widget()
        self.create_event_id_widget()
        self.create_telid_widget()
        self.create_channel_widget()
        self.create_extractor_widgets()
        self.update_extractor_widget_values()

        # Setup layout
        self.layout = layout([
            [self.viewer.layout],
            [self.w_event_index, self.w_event_id, self.w_telid,
             self.w_channel],
            [self.wb_extractor]
        ])

    def start(self):
        self.event_index = 0

    def finish(self):
        curdoc().add_root(self.layout)
        curdoc().title = "Event Viewer"

    @property
    def event_index(self):
        return self._event_index

    @event_index.setter
    def event_index(self, val):
        self._event_index = val
        self.event = self.file_reader.get_event(val, False)

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, val):
        self._event_id = val
        self.event = self.file_reader.get_event(val, True)

    @property
    def telid(self):
        return self._telid

    @telid.setter
    def telid(self, val):
        n_chan = self.event.inst.num_channels[val]
        if self.channel is None or self.channel >= n_chan:
            self.channel = 0

        tels = list(self.event.r0.tels_with_data)
        if val not in tels:
            val = tels[0]
        self._telid = val
        self.viewer.telid = val
        self.update_telid_widget()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, val):
        self._channel = val
        self.viewer.channel = val
        self.update_channel_widget()

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, val):

        # Calibrate
        self.r1.calibrate(val)
        self.dl0.reduce(val)
        self.dl1.calibrate(val)

        self._event = val

        self.viewer.event = val

        self._event_index = val.count
        self._event_id = val.r0.event_id
        self.update_event_index_widget()
        self.update_event_id_widget()

        self._telid = self.viewer.telid
        self.update_telid_widget()

        self._channel = self.viewer.channel
        self.update_channel_widget()

    @property
    def extractor(self):
        return self._extractor

    @extractor.setter
    def extractor(self, val):
        self._extractor = val
        kwargs = dict(config=self.config, tool=self)
        self.dl1 = CameraDL1Calibrator(extractor=val, **kwargs)
        self.dl1.calibrate(self.event)
        self.viewer.refresh()

    def create_event_index_widget(self):
        index_vals = [str(i) for i in range(self.file_reader.num_events)]
        self.w_event_index = Select(title="Event Index:", value='',
                                    options=index_vals)
        self.w_event_index.on_change('value',
                                     self.on_event_index_widget_change)

    def update_event_index_widget(self):
        self.w_event_index.value = str(self.event_index)

    def on_event_index_widget_change(self, attr, old, new):
        if self.event_index != int(self.w_event_index.value):
            self.event_index = int(self.w_event_index.value)

    def create_event_id_widget(self):
        id_vals = [str(i) for i in self.file_reader.event_id_list]
        self.w_event_id = Select(title="Event ID:", value='',
                                 options=id_vals)
        self.w_event_id.on_change('value', self.on_event_id_widget_change)

    def update_event_id_widget(self):
        self.w_event_id.value = str(self.event_id)

    def on_event_id_widget_change(self, attr, old, new):
        if self.event_id != int(self.w_event_id.value):
            self.event_id = int(self.w_event_id.value)

    def create_telid_widget(self):
        self.w_telid = Select(title="Telescope:", value="", options=[])
        self.w_telid.on_change('value', self.on_telid_widget_change)

    def update_telid_widget(self):
        tels = [str(t) for t in self.event.r0.tels_with_data]
        self.w_telid.options = tels
        self.w_telid.value = str(self.telid)

    def on_telid_widget_change(self, attr, old, new):
        if self.telid != int(self.w_telid.value):
            self.telid = int(self.w_telid.value)

    def create_channel_widget(self):
        self.w_channel = Select(title="Channel:", value="", options=[])
        self.w_channel.on_change('value', self.on_channel_widget_change)

    def update_channel_widget(self):
        n_chan = self.event.inst.num_channels[self.telid]
        channels = [str(c) for c in range(n_chan)]
        self.w_channel.options = channels
        self.w_channel.value = str(self.channel)

    def on_channel_widget_change(self, attr, old, new):
        if self.channel != int(self.w_channel.value):
            self.channel = int(self.w_channel.value)

    def create_extractor_widgets(self):
        self.w_dl1_dict = dict(
            extractor=Select(title="Extractor:", value='', width=5,
                             options=ChargeExtractorFactory.subclass_names),
            window_width=TextInput(title="Window Width:", value=''),
            window_start=TextInput(title="Window Start:", value=''),
            window_shift=TextInput(title="Window Shift:", value=''),
            sig_amp_cut_HG=TextInput(title="Significant Amplitude Cut (HG):",
                                     value=''),
            sig_amp_cut_LG=TextInput(title="Significant Amplitude Cut (LG):",
                                     value=''),
            lwt=TextInput(title="Local Pixel Weight:", value=''))

        for key, val in self.w_dl1_dict.items():
            val.on_change('value', self.on_extractor_widget_change)

        self.wb_extractor = widgetbox(PreText(text="Charge Extractor "
                                                   "Configuration"),
                                      self.w_dl1_dict['extractor'],
                                      self.w_dl1_dict['window_width'],
                                      self.w_dl1_dict['window_start'],
                                      self.w_dl1_dict['window_shift'],
                                      self.w_dl1_dict['sig_amp_cut_HG'],
                                      self.w_dl1_dict['sig_amp_cut_LG'],
                                      self.w_dl1_dict['lwt'])

    def update_extractor_widget_values(self):
        for key, val in self.w_dl1_dict.items():
            if key == 'extractor':
                key = 'name'
            try:
                val.value = str(getattr(self.extractor, key))
            except AttributeError:
                val.value = ''

    def on_extractor_widget_change(self, attr, old, new):
        if self.event:
            if not self._updating_extractor:
                self._updating_extractor = True
                cmdline = []
                for key, val in self.w_dl1_dict.items():
                    if val.value:
                        cmdline.append('--{}'.format(key))
                        cmdline.append(val.value)
                self.parse_command_line(cmdline)
                kwargs = dict(config=self.config, tool=self)
                extractor_factory = ChargeExtractorFactory(**kwargs)
                extractor_class = extractor_factory.get_class()
                self.extractor = extractor_class(**kwargs)
                self.update_extractor_widget_values()
                self._updating_extractor = False


exe = BokehFileViewer()
exe.run()
