from bokeh.io import curdoc
from bokeh.layouts import layout, widgetbox
from bokeh.plotting import figure
from traitlets import Dict, List
from bokeh.models import Select, ColumnDataSource, palettes, \
    RadioGroup, CheckboxGroup, Legend, Div, Span
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.io import CameraGeometry
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.visualization.bokeh import CameraDisplay
from targetpipe.io.pixels import get_neighbours_2d
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitter
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from time import sleep


class Camera(CameraDisplay):
    name = 'Camera'

    def __init__(self, parent, neighbours2d, title, geometry=None, image=None):
        self._active_pixel = None
        self.parent = parent
        self.neighbours2d = neighbours2d
        fig = figure(title=title, plot_width=600, plot_height=400, tools="",
                     toolbar_location=None, outline_line_color='#595959')
        super().__init__(geometry=geometry, image=image, fig=fig)

    def enable_pixel_picker(self, _=None):
        super().enable_pixel_picker(1)

    def highlight_pixels(self):
        alpha = np.zeros(self._n_pixels)
        color = np.full(self._n_pixels, 'black', dtype=np.object_)
        if self.active_pixels:
            pix = self.active_pixels[0]
            if self.neighbours2d:
                nei = self.neighbours2d[pix]
                nei = nei[~np.isnan(nei)].astype(np.int)
                alpha[nei] = 1
                color[nei] = 'white'
            alpha[pix] = 1
            color[pix] = 'red'
            self.cdsource.data['outline_alpha'] = alpha
            self.cdsource.data['outline_color'] = color

    def _on_pixel_click(self, pix_id):
        super()._on_pixel_click(pix_id)
        self.parent.active_pixel = pix_id

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self.active_pixels = [val]


class StageViewer(Component):
    name = 'StageViewer'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of TF across cells

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
        self._active_pixel = 0

        self.figures = None
        self.cdsources = None
        self.lines = None

        self.x = None
        self.stages = None
        self.neighbours2d = None
        self.stage_list = None
        self.active_stages = None
        self.w_pulse = None
        self.w_integration = None

        self.cb = None
        self.pulsewin1 = []
        self.pulsewin2 = []
        self.intwin1 = []
        self.intwin2 = []

        self.layout = None

    def create(self, neighbours2d, stage_list):

        self.neighbours2d = neighbours2d
        self.stage_list = stage_list

        palette = palettes.Set1[9]

        self.figures = []
        self.cdsources = []
        legend_list = []
        self.lines = defaultdict(list)
        for i in range(9):
            fig = figure(plot_width=400, plot_height=200, tools="",
                         toolbar_location=None, outline_line_color='#595959')
            cdsource_d = dict(x=[])
            for stage in self.stage_list:
                cdsource_d[stage] = []
            cdsource = ColumnDataSource(data=cdsource_d)
            for j, stage in enumerate(self.stage_list):
                color = palette[j % len(palette)]
                l = fig.line(source=cdsource, x='x', y=stage, color=color)
                if not j == 0 and not j == 7:
                    l.visible = False
                self.lines[stage].append(l)
                if i == 2:
                    legend_list.append((stage, [l]))

            self.figures.append(fig)
            self.cdsources.append(cdsource)

            self.pulsewin1.append(Span(location=0, dimension='height',
                                  line_color='red', line_dash='dotted'))
            self.pulsewin2.append(Span(location=0, dimension='height',
                                  line_color='red', line_dash='dotted'))
            self.intwin1.append(Span(location=0, dimension='height',
                                line_color='green', line_dash='dotted'))
            self.intwin2.append(Span(location=0, dimension='height',
                                line_color='green', line_dash='dotted'))
            fig.add_layout(self.pulsewin1[i])
            fig.add_layout(self.pulsewin2[i])
            fig.add_layout(self.intwin1[i])
            fig.add_layout(self.intwin2[i])

            if i == 2:
                legend = Legend(items=legend_list, location=(0, 0),
                                background_fill_alpha=0,
                                label_text_color='green')
                fig.add_layout(legend, 'right')

        self.cb = CheckboxGroup(labels=self.stage_list, active=[0, 7])
        self.cb.on_click(self._on_checkbox_select)
        self.active_stages = [self.stage_list[i] for i in self.cb.active]

        figures = layout([
            [self.figures[0], self.figures[1], self.figures[2]],
            [self.figures[3], self.figures[4], self.figures[5]],
            [self.figures[6], self.figures[7], self.figures[8]]
        ])
        self.layout = layout([
            [self.cb, figures]
        ])

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            self.update_stages(self.x, self.stages,
                               self.w_pulse, self.w_integration)

    def _get_neighbour_pixel(self, i):
        pixel = self.neighbours2d[self.active_pixel].ravel()[i]
        if np.isnan(pixel):
            return None
        return int(pixel)

    def update_stages(self, x, stages, w_pulse, w_integration):
        self.x = x
        self.stages = stages
        self.w_pulse = w_pulse
        self.w_integration = w_integration
        for i, cdsource in enumerate(self.cdsources):
            pixel = self._get_neighbour_pixel(i)
            if pixel is None:
                cdsource_d = dict(x=[])
                for stage, values in self.stages.items():
                    cdsource_d[stage] = []
                cdsource.data = cdsource_d
            else:
                cdsource_d = dict(x=x)
                for stage, values in self.stages.items():
                    if values.ndim == 2:
                        cdsource_d[stage] = values[pixel]
                    elif values.ndim == 1:
                        cdsource_d[stage] = values
                    else:
                        self.log.error("Too many dimensions in stage values")
                cdsource.data = cdsource_d
            self.pulsewin1[i].set(location=self.w_pulse[0])
            self.pulsewin2[i].set(location=self.w_pulse[1])
            self.intwin1[i].set(location=self.w_integration[0])
            self.intwin2[i].set(location=self.w_integration[1])
        sleep(0.1)
        self._update_yrange()

    def toggle_stage(self, stage, value):
        lines = self.lines[stage]
        for l in lines:
            l.visible = value

    def _on_checkbox_select(self, active):
        self.active_stages = [self.stage_list[i] for i in self.cb.active]
        for stage in self.stage_list:
            if stage in self.active_stages:
                self.toggle_stage(stage, True)
            else:
                self.toggle_stage(stage, False)
        self._update_yrange()

    def _update_yrange(self):
        for fig, cdsource in zip(self.figures, self.cdsources):
            min_l = []
            max_l = []
            for stage in self.active_stages:
                array = cdsource.data[stage]
                min_l.append(min(array))
                max_l.append(max(array))
            min_ = min(min_l)
            max_ = max(max_l)
            fig.y_range.start = min_
            fig.y_range.end = max_


class SpectrumViewer(Component):
    name = 'SpectrumViewer'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of TF across cells

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
        self._active_pixel = 0

        self.figures = None
        self.cdsources = None

        self.x = None
        self.stages = None
        self.neighbours2d = None
        self.peak_area = None
        self.peak_height = None
        self.fitter = CHECMFitter(config, tool, apply_fit=False)

        self.radio = None

        self.layout = None

    def create(self, neighbours2d, peak_area, peak_height):

        self.neighbours2d = neighbours2d
        self.peak_area = peak_area
        self.peak_height = peak_height

        self.figures = []
        self.cdsources = []
        for i in range(9):
            fig = figure(plot_width=400, plot_height=200, tools="",
                         toolbar_location=None, outline_line_color='#595959')
            cdsource_d = dict(left=[], right=[], bottom=[], top=[])
            cdsource = ColumnDataSource(data=cdsource_d)
            fig.quad(bottom='bottom', left='left', right='right',
                     top='top', source=cdsource, alpha=0.5)
            self.figures.append(fig)
            self.cdsources.append(cdsource)

        self.radio = RadioGroup(labels=['area', 'height'], active=0)
        self.radio.on_click(self._on_radio_select)

        self.build_histogram()

        figures = layout([
            [self.figures[0], self.figures[1], self.figures[2]],
            [self.figures[3], self.figures[4], self.figures[5]],
            [self.figures[6], self.figures[7], self.figures[8]]
        ])
        self.layout = layout([
            [self.radio, figures]
        ])

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            self.build_histogram()

    def _get_neighbour_pixel(self, i):
        pixel = self.neighbours2d[self.active_pixel].ravel()[i]
        if np.isnan(pixel):
            return None
        return int(pixel)

    def build_histogram(self):
        for i, cdsource in enumerate(self.cdsources):
            pixel = self._get_neighbour_pixel(i)
            if pixel is None:
                area = [-1]
                height = [-1]
            else:
                area = self.peak_area[:, pixel]
                height = self.peak_height[:, pixel]
            if self.radio.active == 0:
                self.fitter.apply(area)
                hist = self.fitter.hist
                edges = self.fitter.edges
            else:
                self.fitter.apply(height, True)
                hist = self.fitter.hist
                edges = self.fitter.edges

            zeros = np.zeros(edges.size - 1)
            left = edges[:-1]
            right = edges[1:]
            cdsource_d = dict(left=left, right=right, bottom=zeros, top=hist)
            cdsource.data = cdsource_d

    def _on_radio_select(self, _):
        self.build_histogram()


class FitViewer(Component):
    name = 'FitViewer'

    def __init__(self, config, tool, **kwargs):
        """
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
        self._active_pixel = 0

        self.figure = None
        self.cdsource = None
        self.cdsource_f = None

        self.x = None
        self.stages = None
        self.neighbours2d = None
        self.peak_area = None
        self.peak_height = None
        self.fitter = CHECMFitter(config, tool, apply_fit=True)
        self.fits = None
        self.fit_labels = None

        self.radio = None
        self.cb = None

        self.layout = None

    def create(self, peak_area, peak_height):

        self.peak_area = peak_area
        self.peak_height = peak_height

        title = "Fit Viewer"
        fig = figure(title=title, plot_width=400, plot_height=400, tools="",
                     toolbar_location=None, outline_line_color='#595959')
        cdsource_d = dict(left=[], right=[], bottom=[], top=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)
        fig.quad(bottom='bottom', left='left', right='right',
                 top='top', source=self.cdsource, alpha=0.5)

        cdsource_d_fit = dict(x=[], fit=[])
        self.fit_labels = ['fit']
        for subfit in self.fitter.subfit_labels:
            cdsource_d_fit[subfit] = []
        self.cdsource_f = ColumnDataSource(data=cdsource_d_fit)
        l1 = fig.line('x', 'fit', source=self.cdsource_f, color='yellow')
        self.fits = dict(fit=l1)
        for i, subfit in enumerate(self.fitter.subfit_labels):
            l = fig.line('x', subfit, source=self.cdsource_f, color='red')
            l.visible = False
            self.fits[subfit] = l

        self.fit_labels.extend(self.fitter.subfit_labels)
        self.cb = CheckboxGroup(labels=self.fit_labels, active=[0])
        self.cb.on_click(self._on_checkbox_select)

        self.radio = RadioGroup(labels=['area', 'height'], active=0)
        self.radio.on_click(self._on_radio_select)

        self.build_fit()

        figures = layout([
            [fig],
        ])
        widgets = widgetbox([self.radio, self.cb])
        self.layout = layout([
            [figures, widgets]
        ])

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            self.build_fit()

    def build_fit(self):
        area = self.peak_area[:, self.active_pixel]
        height = self.peak_height[:, self.active_pixel]
        try:
            if self.radio.active == 0:
                self.fitter.apply(area)
            else:
                self.fitter.apply(height, True)
        except RuntimeError:
            self.log.warning("Pixel {} could not be fitted"
                             .format(self.active_pixel))
        hist = self.fitter.hist
        edges = self.fitter.edges
        zeros = np.zeros(edges.size - 1)
        left = edges[:-1]
        right = edges[1:]
        cdsource_d = dict(left=left, right=right, bottom=zeros, top=hist)
        self.cdsource.data = cdsource_d

        cdsource_d_fit = dict(x=self.fitter.fit_x, fit=self.fitter.fit)
        for subfit, values in self.fitter.subfits.items():
            cdsource_d_fit[subfit] = values
        self.cdsource_f.data = cdsource_d_fit

    def _on_radio_select(self, _):
        self.build_fit()

    def _on_checkbox_select(self, active):
        self.active_fits = [self.fit_labels[i] for i in self.cb.active]
        for fit, line in self.fits.items():
            if fit in self.active_fits:
                line.visible = True
            else:
                line.visible = False


class BokehSPE(Tool):
    name = "BokehSPE"
    description = "Interactively explore the steps in obtaining and fitting " \
                  "SPE spectrum"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        brightness='CHECMFitter.brightness'
                        ))
    flags = Dict(dict(no_fit=({'CHECMFitter': {'apply_fit': False}},
                              'Dont apply the fit'),
                      ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    CHECMFitter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = None
        self._event_index = None
        self._event_id = None
        self._active_pixel = None

        self.w_event_index = None
        self.layout = None

        self.reader = None
        self.r1 = None
        self.dl0 = None

        self.n_events = None
        self.n_pixels = None
        self.n_samples = None

        self.cleaner = None
        self.extractor = None
        self.fitter = None

        self.neighbours2d = None
        self.stage_names = None

        self.p_camera_area = None
        self.p_camera_gain = None
        self.p_stage_viewer = None
        self.p_spectrum_viewer = None
        self.p_fit_viewer = None

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
        self.fitter = CHECMFitter(**kwargs)

        self.n_events = self.reader.num_events
        first_event = self.reader.get_event(0)
        telid = list(first_event.r0.tels_with_data)[0]
        geom = CameraGeometry.guess(*first_event.inst.pixel_pos[telid],
                                    first_event.inst.optical_foclen[telid])
        self.neighbours2d = get_neighbours_2d(geom.pix_x, geom.pix_y)

        # Get stage names
        r0 = first_event.r0.tel[telid].adc_samples[0]
        self.n_pixels, self.n_samples = r0.shape
        self.cleaner.apply(r0)
        self.stage_names = sorted(list(self.cleaner.stages.keys()))

        # Init Plots
        self.p_camera_area = Camera(self, self.neighbours2d, "Area", geom)
        self.p_camera_gain = Camera(self, self.neighbours2d, "Gain", geom)
        self.p_stage_viewer = StageViewer(**kwargs)
        self.p_spectrum_viewer = SpectrumViewer(**kwargs)
        self.p_fit_viewer = FitViewer(**kwargs)

    def start(self):
        # Prepare storage array
        area = np.zeros((self.n_events, self.n_pixels))
        height = np.zeros((self.n_events, self.n_pixels))
        global_ = np.zeros((self.n_events, self.n_samples))
        self.gain = np.zeros(self.n_pixels)

        source = self.reader.read()
        desc = "Looping through file"
        with tqdm(total=self.n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)
                self.dl0.reduce(event)

                telid = list(event.r0.tels_with_data)[0]
                dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

                # Perform CHECM Waveform Cleaning
                sb_sub_wf, t0 = self.cleaner.apply(dl0)

                # Perform CHECM Charge Extraction
                peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)

                area[index] = peak_area
                height[index] = peak_height
                global_[index] = np.mean(dl0, axis=0)

        desc = "Extracting gain of pixels"
        with tqdm(total=self.n_pixels, desc=desc) as pbar:
            for pix in range(self.n_pixels):
                pbar.update(1)
                try:
                    self.fitter.apply(area[:, pix])
                except RuntimeError:
                    self.log.warning("Pixel {} could not be fitted".format(pix))
                    continue
                self.gain[pix] = self.fitter.gain

        # Setup Plots
        self.p_camera_area.enable_pixel_picker()
        self.p_camera_area.add_colorbar()
        self.p_camera_gain.enable_pixel_picker()
        self.p_camera_gain.add_colorbar()
        self.p_stage_viewer.create(self.neighbours2d, self.stage_names)
        self.p_spectrum_viewer.create(self.neighbours2d, area, height)
        self.p_fit_viewer.create(area, height)

        # Setup widgets
        self.create_event_index_widget()
        self.event_index = 0

        # Get bokeh layouts
        l_camera_area = self.p_camera_area.layout
        l_camera_gain = self.p_camera_gain.layout
        l_stage_viewer = self.p_stage_viewer.layout
        l_spectrum_viewer = self.p_spectrum_viewer.layout
        l_fit_viewer = self.p_fit_viewer.layout

        # Setup layout
        self.layout = layout([
            [self.w_event_index],
            [l_camera_area, l_camera_gain, l_fit_viewer],
            [Div(text="Stage Viewer")],
            [l_stage_viewer],
            [Div(text="Spectrum Viewer")],
            [l_spectrum_viewer]
        ])

    def finish(self):
        curdoc().add_root(self.layout)
        curdoc().title = "Event Viewer"

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, val):

        # Calibrate
        self.r1.calibrate(val)
        self.dl0.reduce(val)

        self._event = val

        telid = list(val.r0.tels_with_data)[0]
        self.p_camera_area._telid = telid
        self.p_camera_gain._telid = telid

        self._event_index = val.count
        self._event_id = val.r0.event_id
        self.update_event_index_widget()

        dl0 = np.copy(val.dl0.tel[telid].pe_samples[0])
        n_pixels, n_samples = dl0.shape

        # Perform CHECM Waveform Cleaning
        sb_sub_wf, t0 = self.cleaner.apply(dl0)
        stages = self.cleaner.stages
        pw_l = self.cleaner.pw_l
        pw_r = self.cleaner.pw_r

        # Perform CHECM Charge Extraction
        peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)
        iw_l = self.extractor.iw_l
        iw_r = self.extractor.iw_r

        self.p_camera_area.image = peak_area
        self.p_camera_gain.image = self.gain
        self.p_stage_viewer.update_stages(np.arange(n_samples), stages,
                                          [pw_l, pw_r], [iw_l, iw_r])

    @property
    def event_index(self):
        return self._event_index

    @event_index.setter
    def event_index(self, val):
        self._event_index = val
        self.event = self.reader.get_event(val, False)

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            self.p_camera_area.active_pixel = val
            self.p_camera_gain.active_pixel = val
            self.p_stage_viewer.active_pixel = val
            self.p_spectrum_viewer.active_pixel = val
            self.p_fit_viewer.active_pixel = val

    def create_event_index_widget(self):
        index_vals = [str(i) for i in range(self.reader.num_events)]
        self.w_event_index = Select(title="Event Index:", value='',
                                    options=index_vals)
        self.w_event_index.on_change('value',
                                     self.on_event_index_widget_change)

    def update_event_index_widget(self):
        self.w_event_index.value = str(self.event_index)

    def on_event_index_widget_change(self, attr, old, new):
        if self.event_index != int(self.w_event_index.value):
            self.event_index = int(self.w_event_index.value)


exe = BokehSPE()
exe.run()
