import numpy as np
from bokeh.layouts import layout, column
from bokeh.models import Select, Span
from traitlets import Int
from ctapipe.core import Component
from ctapipe.instrument import CameraGeometry
from targetpipe.visualization.bokeh import CameraDisplay, WaveformDisplay


class EventViewerCamera(CameraDisplay):
    def __init__(self, event_viewer, fig=None):
        self._event = None
        self._view = 'r0'
        self._telid = None
        self._channel = 0
        self._time = 0
        super().__init__(fig=fig)

        self._view_options = ['r0', 'r1', 'dl0', 'dl1', 'peakpos', 'cleaned']
        self.w_view = None
        self._geom_tel = None
        self._geom_url = None

        self.event_viewer = event_viewer

    def _reset(self):
        self.reset_pixels()
        self.event_viewer.change_time(0)

    def _set_image(self):
        e = self.event
        v = self.view
        t = self.telid
        c = self.channel
        time = self.time
        if e:
            tels = list(e.r0.tels_with_data)
            if t is None:
                t = tels[0]
            if v == 'r0':
                samples = e.r0.tel[t].adc_samples
                if samples is None:
                    self.image = None
                else:
                    self.image = samples[c, :, time]
                self.fig.title.text = 'R0 ADC Samples (T = {})'.format(time)
                if self.cb:
                    self.cb.title = "ADC"
            elif v == 'r1':
                samples = e.r1.tel[t].pe_samples
                if samples is None:
                    self.image = None
                else:
                    self.image = samples[c, :, time]
                self.fig.title.text = 'R1 p.e. Samples (T = {})'.format(time)
                if self.cb:
                    self.cb.title = "p.e."
            elif v == 'dl0':
                samples = e.dl0.tel[t].pe_samples
                if samples is None:
                    self.image = None
                else:
                    self.image = samples[c, :, time]
                self.fig.title.text = 'DL0 p.e. Samples (T = {})'.format(time)
                if self.cb:
                    self.cb.title = "p.e."
            elif v == 'dl1':
                self.image = e.dl1.tel[t].image[c, :]
                self.fig.title.text = 'DL1 p.e. Image'
                if self.cb:
                    self.cb.title = "p.e."
            elif v == 'peakpos':
                peakpos = e.dl1.tel[t].peakpos
                if peakpos is None:
                    self.image = np.zeros(self.image.shape)
                else:
                    self.image = peakpos[c, :]
                self.fig.title.text = 'DL1 Peakpos Image'
                if self.cb:
                    self.cb.title = "ns"
            elif v == 'cleaned':
                samples = e.dl1.tel[t].cleaned
                if samples is None:
                    self.image = None
                else:
                    self.image = samples[c, :, time]
                self.fig.title.text = 'Cleaned p.e. Samples (T = {})'\
                    .format(time)
                if self.cb:
                    self.cb.title = "p.e."
            else:
                raise ValueError("No view configuration set up "
                                 "for: {}".format(v))
        else:
            self.event_viewer.log.warning("No event has been provided")

    def _update_geometry(self):
        e = self.event
        t = self.telid
        if e:
            url = e.meta['input']
            # Check if geom actually needs to be changed
            if not (t == self._geom_tel and url == self._geom_url):
                self.geom = e.inst.subarray.tel[t].camera
                self._geom_tel = t
                self._geom_url = url
        else:
            self.event_viewer.log.warning("No event has been provided")

    def refresh(self):
        self._set_image()

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, val):
        self._event = val
        self._update_geometry()
        self._set_image()

    def change_event(self, event, telid):
        if self.event:  # Only reset when an event exists
            self._reset()
        self._telid = telid
        self.event = event

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, val):
        if val not in self._view_options:
            raise ValueError("View is not valid: {}".format(val))
        self._view = val
        self._set_image()

    @property
    def telid(self):
        return self._telid

    @telid.setter
    def telid(self, val):
        if self.event:  # Only reset when an event exists
            self._reset()
        self._telid = val
        self._update_geometry()
        self._set_image()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, val):
        self._channel = val
        self._set_image()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._time = int(val)
        self._set_image()

    def _on_pixel_click(self, pix_id):
        super()._on_pixel_click(pix_id)
        ai = self.active_index
        self.event_viewer.waveforms[ai].pixel = pix_id

    def create_view_widget(self):
        self.w_view = Select(title="View:", value="", options=[], width=5)
        self.w_view.on_change('value', self.on_view_widget_change)
        self.layout = column([self.w_view, self.layout])

    def update_view_widget(self):
        self.w_view.options = self._view_options
        self.w_view.value = self.view

    def on_view_widget_change(self, attr, old, new):
        if self.view != self.w_view.value:
            self.view = self.w_view.value


class EventViewerWaveform(WaveformDisplay):
    def __init__(self, event_viewer, fig=None):
        self._event = None
        self._view = 'r0'
        self._telid = None
        self._channel = 0
        self._pixel = 0
        super().__init__(fig=fig)
        self._draw_integration_window()

        self._view_options = ['r0', 'r1', 'dl0', 'cleaned']
        self.w_view = None

        self.event_viewer = event_viewer

    def _reset(self):
        for wav in self.event_viewer.waveforms:
            wav.pixel = 0

    def _set_waveform(self):
        e = self.event
        v = self.view
        t = self.telid
        c = self.channel
        p = self.pixel
        if e:
            if v == 'r0':
                samples = e.r0.tel[t].adc_samples
                if samples is None:
                    self.waveform = None
                else:
                    self.waveform = samples[c, p]
                self.fig.title.text = 'R0 ADC Samples (Pixel = {})'.format(p)
                self.fig.yaxis.axis_label = 'ADC'
            elif v == 'r1':
                samples = e.r1.tel[t].pe_samples
                if samples is None:
                    self.waveform = None
                else:
                    self.waveform = samples[c, p]
                self.fig.title.text = 'R1 p.e. Samples (Pixel = {})'.format(p)
                self.fig.yaxis.axis_label = 'p.e.'
            elif v == 'dl0':
                samples = e.dl0.tel[t].pe_samples
                if samples is None:
                    self.waveform = None
                else:
                    self.waveform = samples[c, p]
                self.fig.title.text = 'DL0 p.e. Samples (Pixel = {})'.format(p)
                self.fig.yaxis.axis_label = 'p.e.'
            elif v == 'cleaned':
                samples = e.dl1.tel[t].cleaned
                if samples is None:
                    self.waveform = None
                else:
                    self.waveform = samples[c, p]
                self.fig.title.text = 'Cleaned p.e. Samples (Pixel = {})'\
                    .format(p)
                self.fig.yaxis.axis_label = 'p.e.'
            else:
                raise ValueError("No view configuration set up "
                                 "for: {}".format(v))
        else:
            self.event_viewer.log.warning("No event has been provided")

    def _draw_integration_window(self):
        self.intwin1 = Span(location=0, dimension='height',
                            line_color='green', line_dash='dotted')
        self.intwin2 = Span(location=0, dimension='height',
                            line_color='green', line_dash='dotted')
        self.fig.add_layout(self.intwin1)
        self.fig.add_layout(self.intwin2)

    def _set_integration_window(self):
        e = self.event
        t = self.telid
        c = self.channel
        p = self.pixel
        if e:
            if e.dl1.tel[t].extracted_samples is not None:
                # Get Windows
                windows = e.dl1.tel[t].extracted_samples[c, p]
                length = np.sum(windows)
                start = np.argmax(windows)
                end = start + length - 1
                self.intwin1.location = start
                self.intwin2.location = end
        else:
            self.event_viewer.log.warning("No event has been provided")

    def refresh(self):
        self._set_waveform()
        self._set_integration_window()

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, val):
        self._event = val
        self._set_waveform()
        self._set_integration_window()

    def change_event(self, event, telid):
        if self.event:  # Only reset when an event exists
            self._reset()
        self._telid = telid
        self.event = event

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, val):
        if val not in self._view_options:
            raise ValueError("View is not valid: {}".format(val))
        self._view = val
        self._set_waveform()
        self._set_integration_window()

    @property
    def telid(self):
        return self._telid

    @telid.setter
    def telid(self, val):
        if self.event:  # Only reset when an event exists
            self._reset()
        self._telid = val
        self._set_waveform()
        self._set_integration_window()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, val):
        self._channel = val
        self._set_waveform()
        self._set_integration_window()

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, val):
        self._pixel = val
        self._set_waveform()
        self._set_integration_window()

    def _on_waveform_click(self, time):
        super()._on_waveform_click(time)
        self.event_viewer.change_time(time)

    def create_view_widget(self):
        self.w_view = Select(title="View:", value="", options=[], width=5)
        self.w_view.on_change('value', self.on_view_widget_change)
        self.layout = column([self.w_view, self.layout])

    def update_view_widget(self):
        self.w_view.options = self._view_options
        self.w_view.value = self.view

    def on_view_widget_change(self, attr, old, new):
        if self.view != self.w_view.value:
            self.view = self.w_view.value


class EventViewer(Component):
    name = 'EventViewer'

    num_cameras = Int(1, help="Number of CameraDisplays to "
                              "create").tag(config=True)
    num_waveforms = Int(2, help="Number of WaveformDisplays to "
                                "create").tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Parent class for the r1 calibrators. Fills the r1 container.

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

        self._event = None
        self._view = 'r0'
        self._telid = None
        self._channel = 0

        self.cameras = []
        self.camera_layouts = []
        self.waveforms = []
        self.waveform_layouts = []

        self.layout = None

    def create(self):
        for icam in range(self.num_cameras):
            cam = EventViewerCamera(self)
            cam.enable_pixel_picker(self.num_waveforms)
            cam.create_view_widget()
            cam.update_view_widget()
            cam.add_colorbar()

            # # Sync sources, so they all change at the same time
            # if icam == 0:
            #     self.pixel_cdsource = cam.pixel_cdsource
            # else:
            #     cam.pixel_cdsource = self.pixel_cdsource

            self.cameras.append(cam)
            self.camera_layouts.append(cam.layout)

        for iwav in range(self.num_waveforms):
            wav = EventViewerWaveform(self)
            active_color = self.cameras[0].active_colors[iwav]
            wav.fig.select(name='line')[0].glyph.line_color = active_color
            wav.enable_time_picker()
            wav.create_view_widget()
            wav.update_view_widget()

            self.waveforms.append(wav)
            self.waveform_layouts.append(wav.layout)

        self.layout = layout([
            [column(self.camera_layouts), column(self.waveform_layouts)],
        ])

    def enable_automatic_index_increment(self):
        for cam in self.cameras:
            cam.automatic_index_increment = True

    def change_time(self, time):
        # TODO: Would a cdsource be better (with an on_change) for better sync
        for wav in self.waveforms:
            if wav.active_time != time:
                wav.active_time = time
        for camera in self.cameras:
            camera.time = self.waveforms[0].active_time

    def sub_event_viewer_generator(self):
        for camera in self.cameras:
            yield camera
        for waveform in self.waveforms:
            yield waveform

    def refresh(self):
        for sub in self.sub_event_viewer_generator():
            sub.refresh()

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, val):
        if self._event != val:
            self._event = val
            tels = list(val.r0.tels_with_data)
            if self.telid not in tels:
                self._telid = tels[0]
            for sub in self.sub_event_viewer_generator():
                sub.change_event(val, self.telid)

    @property
    def telid(self):
        return self._telid

    @telid.setter
    def telid(self, val):
        if self._telid != val:
            self._telid = val
            for sub in self.sub_event_viewer_generator():
                sub.telid = val

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, val):
        if self._channel != val:
            self._channel = val
            for sub in self.sub_event_viewer_generator():
                sub.channel = val
