from bokeh.io import curdoc
from bokeh.layouts import widgetbox, layout
from bokeh.models import Select, TextInput, PreText, Button
from traitlets import Dict, List

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.core import Tool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from ctapipe.image.waveform_cleaning import WaveformCleanerFactory
from ctapipe.io.eventfilereader import EventFileReaderFactory
from targetpipe.plots.event_viewer import EventViewer


class BokehFileViewer(Tool):
    name = "BokehFileViewer"
    description = "Interactively explore an event file"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        pe='CameraR1CalibratorFactory.pe_path',
                        ff='CameraR1CalibratorFactory.ff_path',
                        extractor='ChargeExtractorFactory.extractor',
                        extractor_t0='ChargeExtractorFactory.t0',
                        extractor_window_width='ChargeExtractorFactory.'
                                               'window_width',
                        extractor_window_shift='ChargeExtractorFactory.'
                                               'window_shift',
                        extractor_sig_amp_cut_HG='ChargeExtractorFactory.'
                                                 'sig_amp_cut_HG',
                        extractor_sig_amp_cut_LG='ChargeExtractorFactory.'
                                                 'sig_amp_cut_LG',
                        extractor_lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        cleaner='WaveformCleanerFactory.cleaner',
                        cleaner_t0='WaveformCleanerFactory.t0',
                        ))
    flags = Dict(dict(id=({'DisplayDL1Calib': {'use_event_index': True}},
                          'event_index will obtain an event using '
                          'event_id instead of index.')
                      ))
    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraR1CalibratorFactory,
                    CameraDL1Calibrator,
                    WaveformCleanerFactory
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = None
        self._event_index = None
        self._event_id = None
        self._telid = None
        self._channel = None
        self._extractor = None
        self._cleaner = None

        self.w_next_event = None
        self.w_previous_event = None
        self.w_event_index = None
        self.w_event_id = None
        self.w_goto_event_index = None
        self.w_goto_event_id = None
        self.w_telid = None
        self.w_channel = None
        self.w_dl1_dict = None
        self.wb_extractor = None
        self.layout = None

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.viewer = None

        self._updating_dl1 = False

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        self._extractor = extractor_class(**kwargs)

        cleaner_factory = WaveformCleanerFactory(**kwargs)
        cleaner_class = cleaner_factory.get_class()
        self._cleaner = cleaner_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=self.cleaner,
                                       **kwargs)

        self.viewer = EventViewer(**kwargs)

        # Setup widgets
        self.viewer.create()
        self.viewer.enable_automatic_index_increment()
        self.create_previous_event_widget()
        self.create_next_event_widget()
        self.create_event_index_widget()
        self.create_goto_event_index_widget()
        self.create_event_id_widget()
        self.create_goto_event_id_widget()
        self.create_telid_widget()
        self.create_channel_widget()
        self.create_dl1_widgets()
        self.update_dl1_widget_values()

        # Setup layout
        self.layout = layout([
            [self.viewer.layout],
            [self.w_previous_event, self.w_next_event, self.w_goto_event_index, self.w_goto_event_id],
            [self.w_event_index, self.w_event_id],
            [self.w_telid, self.w_channel],
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
        try:
            self.event = self.reader.get_event(val, False)
        except RuntimeError:
            self.log.warning("Event Index {} does not exist".format(val))

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, val):
        try:
            self.event = self.reader.get_event(val, True)
        except RuntimeError:
            self.log.warning("Event ID {} does not exist".format(val))

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
        self.dl1 = CameraDL1Calibrator(extractor=val,
                                       cleaner=self.cleaner,
                                       **kwargs)
        self.dl1.calibrate(self.event)
        self.viewer.refresh()

    @property
    def cleaner(self):
        return self._cleaner

    @cleaner.setter
    def cleaner(self, val):
        self._cleaner = val
        kwargs = dict(config=self.config, tool=self)
        self.dl1 = CameraDL1Calibrator(extractor=self.extractor,
                                       cleaner=val,
                                       **kwargs)
        self.dl1.calibrate(self.event)
        self.viewer.refresh()

    def create_next_event_widget(self):
        self.w_next_event = Button(label=">", button_type="default", width=50)
        self.w_next_event.on_click(self.on_next_event_widget_click)

    def on_next_event_widget_click(self):
        self.event_index += 1

    def create_previous_event_widget(self):
        self.w_previous_event = Button(label="<", button_type="default", width=50)
        self.w_previous_event.on_click(self.on_previous_event_widget_click)

    def on_previous_event_widget_click(self):
        # TODO don't allow negative
        self.event_index -= 1

    def create_event_index_widget(self):
        self.w_event_index = TextInput(title="Event Index:", value='')

    def update_event_index_widget(self):
        if self.w_event_index:
            self.w_event_index.value = str(self.event_index)

    def create_event_id_widget(self):
        self.w_event_id = TextInput(title="Event ID:", value='')

    def update_event_id_widget(self):
        if self.w_event_id:
            self.w_event_id.value = str(self.event_id)

    def create_goto_event_index_widget(self):
        self.w_goto_event_index = Button(label="GOTO Index", button_type="default", width=100)
        self.w_goto_event_index.on_click(self.on_goto_event_index_widget_click)

    def on_goto_event_index_widget_click(self):
        self.event_index = int(self.w_event_index.value)

    def create_goto_event_id_widget(self):
        self.w_goto_event_id = Button(label="GOTO ID", button_type="default", width=70)
        self.w_goto_event_id.on_click(self.on_goto_event_id_widget_click)

    def on_goto_event_id_widget_click(self):
        self.event_id = int(self.w_event_id.value)

    def create_telid_widget(self):
        self.w_telid = Select(title="Telescope:", value="", options=[])
        self.w_telid.on_change('value', self.on_telid_widget_change)

    def update_telid_widget(self):
        if self.w_telid:
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
        if self.channel:
            n_chan = self.event.inst.num_channels[self.telid]
            channels = [str(c) for c in range(n_chan)]
            self.w_channel.options = channels
            self.w_channel.value = str(self.channel)

    def on_channel_widget_change(self, attr, old, new):
        if self.channel != int(self.w_channel.value):
            self.channel = int(self.w_channel.value)

    def create_dl1_widgets(self):
        self.w_dl1_dict = dict(
            cleaner=Select(title="Cleaner:", value='', width=5,
                           options=WaveformCleanerFactory.subclass_names),
            extractor=Select(title="Extractor:", value='', width=5,
                             options=ChargeExtractorFactory.subclass_names),
            extractor_t0=TextInput(title="T0:", value=''),
            extractor_window_width=TextInput(title="Window Width:", value=''),
            extractor_window_shift=TextInput(title="Window Shift:", value=''),
            extractor_sig_amp_cut_HG=TextInput(title="Significant Amplitude "
                                                     "Cut (HG):", value=''),
            extractor_sig_amp_cut_LG=TextInput(title="Significant Amplitude "
                                                     "Cut (LG):", value=''),
            extractor_lwt=TextInput(title="Local Pixel Weight:", value=''))

        for key, val in self.w_dl1_dict.items():
            val.on_change('value', self.on_dl1_widget_change)

        self.wb_extractor = widgetbox(
            PreText(text="Charge Extractor Configuration"),
            self.w_dl1_dict['cleaner'],
            self.w_dl1_dict['extractor'],
            self.w_dl1_dict['extractor_t0'],
            self.w_dl1_dict['extractor_window_width'],
            self.w_dl1_dict['extractor_window_shift'],
            self.w_dl1_dict['extractor_sig_amp_cut_HG'],
            self.w_dl1_dict['extractor_sig_amp_cut_LG'],
            self.w_dl1_dict['extractor_lwt'])

    def update_dl1_widget_values(self):
        if self.w_dl1_dict:
            for key, val in self.w_dl1_dict.items():
                if 'extractor' in key:
                    if key == 'extractor':
                        key = 'name'
                    else:
                        key = key.replace("extractor_", "")
                    try:
                        val.value = str(getattr(self.extractor, key))
                    except AttributeError:
                        val.value = ''
                elif 'cleaner' in key:
                    if key == 'cleaner':
                        key = 'name'
                    else:
                        key = key.replace("cleaner_", "")
                    try:
                        val.value = str(getattr(self.cleaner, key))
                    except AttributeError:
                        val.value = ''

    def on_dl1_widget_change(self, attr, old, new):
        if self.event:
            if not self._updating_dl1:
                self._updating_dl1 = True
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
                cleaner_factory = WaveformCleanerFactory(**kwargs)
                cleaner_class = cleaner_factory.get_class()
                self.cleaner = cleaner_class(**kwargs)
                self.update_dl1_widget_values()
                self._updating_dl1 = False


exe = BokehFileViewer()
exe.run()
