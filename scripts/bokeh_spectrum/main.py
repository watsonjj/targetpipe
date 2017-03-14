from threading import Thread, Lock

from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.plotting import figure
from traitlets import Dict, List, CaselessStrEnum as CaStEn
from bokeh.models import Select, ColumnDataSource, palettes, \
    RadioGroup, CheckboxGroup, Legend, Div, Span, TableColumn, DataTable, \
    NumberFormatter, TextInput, Button
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.io import CameraGeometry
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from targetpipe.visualization.bokeh import CameraDisplay
from targetpipe.io.pixels import get_neighbours_2d
from targetpipe.calib.camera.waveform_cleaning import CHECMWaveformCleaner
from targetpipe.calib.camera.charge_extractors import CHECMExtractor
from targetpipe.fitting.checm import CHECMFitterSPE, CHECMFitterBright
import numpy as np
from collections import defaultdict
from tqdm import trange, tqdm
from time import sleep, time
from multiprocessing import Pool


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


class FitterWidget(Component):
    name = 'FitterWidget'

    brightness = CaStEn(['spe', 'bright'], 'spe',
                        help='Brightness of run').tag(config=True)

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

        if self.brightness == 'spe':
            self.fitter = CHECMFitterSPE(config, tool)
        else:
            self.fitter = CHECMFitterBright(config, tool)

        self.i_nbins = None
        self.i_min = None
        self.i_max = None
        self.i_initial_l = None
        self.i_lower_l = None
        self.i_upper_l = None

        self.radio = None

        self.layout = None

    def create(self):
        self.i_nbins = TextInput(value=str(self.fitter.nbins), title="Nbins")
        self.i_min = TextInput(value=str(self.fitter.range[0]), title="Min")
        self.i_max = TextInput(value=str(self.fitter.range[1]), title="Max")

        self.i_initial_l = []
        self.i_lower_l = []
        self.i_upper_l = []

        l = [[self.i_nbins, self.i_min, self.i_max]]

        for coeff in self.fitter.coeff_list:
            limit_str = "limit_{}".format(coeff)
            initial = self.fitter.initial[coeff]
            lower = self.fitter.limits[limit_str][0]
            upper = self.fitter.limits[limit_str][1]

            i_initial = TextInput(value=str(initial), title="{}".format(coeff))
            i_lower = TextInput(value=str(lower))
            i_upper = TextInput(value=str(upper))

            self.i_initial_l.append(i_initial)
            self.i_lower_l.append(i_lower)
            self.i_upper_l.append(i_upper)

            l.append([i_initial, i_lower, i_upper])

        self.layout = layout(l)

    def update(self):
        self.i_nbins.value = str(self.fitter.nbins)
        self.i_min.value = str(self.fitter.range[0])
        self.i_max.value = str(self.fitter.range[1])

        inputs = zip(self.i_initial_l, self.i_lower_l, self.i_upper_l)
        for i, (i_initial, i_lower, i_upper) in enumerate(inputs):
            coeff = self.fitter.coeff_list[i]
            limit_str = "limit_{}".format(coeff)
            initial = self.fitter.initial[coeff]
            lower = self.fitter.limits[limit_str][0]
            upper = self.fitter.limits[limit_str][1]

            i_initial.value = str(initial)
            i_lower.value = str(lower)
            i_upper.value = str(upper)

    @staticmethod
    def _convert_string(string):
        if string == 'None':
            val = None
        else:
            val = float(string)
        return val

    def _setup_fitter(self):
        self.fitter.nbins = int(self.i_nbins.value)
        self.fitter.range[0] = self._convert_string(self.i_min.value)
        self.fitter.range[1] = self._convert_string(self.i_max.value)

        inputs = zip(self.i_initial_l, self.i_lower_l, self.i_upper_l)
        for i, (i_initial, i_lower, i_upper) in enumerate(inputs):
            coeff = self.fitter.coeff_list[i]
            l_str = "limit_{}".format(coeff)
            self.fitter.initial[coeff] = self._convert_string(i_initial.value)
            self.fitter.limits[l_str] = (self._convert_string(i_lower.value),
                                         self._convert_string(i_upper.value))

    def fit(self, spectrum):
        self._setup_fitter()
        success = self.fitter.apply(spectrum)
        return success


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
            self.pulsewin1[i].location = self.w_pulse[0]
            self.pulsewin2[i].location = self.w_pulse[1]
            self.intwin1[i].location = self.w_integration[0]
            self.intwin2[i].location = self.w_integration[1]
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
            try:
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
            except ValueError:
                pass  # Edge of camera


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
        self.fits = None
        self.fit_labels = None

        self.cb = None

        self.layout = None

    def create(self, subfit_labels):
        title = "Fit Viewer"
        fig = figure(title=title, plot_width=400, plot_height=400, tools="",
                     toolbar_location=None, outline_line_color='#595959')
        cdsource_d = dict(left=[], right=[], bottom=[], top=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)
        fig.quad(bottom='bottom', left='left', right='right',
                 top='top', source=self.cdsource, alpha=0.5)

        cdsource_d_fit = dict(x=[], fit=[])
        self.fit_labels = ['fit']
        for subfit in subfit_labels:
            cdsource_d_fit[subfit] = []
        self.cdsource_f = ColumnDataSource(data=cdsource_d_fit)
        l1 = fig.line('x', 'fit', source=self.cdsource_f, color='yellow')
        self.fits = dict(fit=l1)
        for i, subfit in enumerate(subfit_labels):
            l = fig.line('x', subfit, source=self.cdsource_f, color='red')
            l.visible = False
            self.fits[subfit] = l

        self.fit_labels.extend(subfit_labels)
        self.cb = CheckboxGroup(labels=self.fit_labels, active=[0])
        self.cb.on_click(self._on_checkbox_select)

        self.layout = layout([
            [fig, self.cb]
        ])

    def update(self, fitter):
        hist = fitter.hist
        edges = fitter.edges
        zeros = np.zeros(edges.size - 1)
        left = edges[:-1]
        right = edges[1:]
        cdsource_d = dict(left=left, right=right, bottom=zeros, top=hist)
        self.cdsource.data = cdsource_d

        cdsource_d_fit = dict(x=fitter.fit_x, fit=fitter.fit)
        for subfit, values in fitter.subfits.items():
            cdsource_d_fit[subfit] = values
        self.cdsource_f.data = cdsource_d_fit

    def _on_checkbox_select(self, active):
        self.active_fits = [self.fit_labels[i] for i in self.cb.active]
        for fit, line in self.fits.items():
            if fit in self.active_fits:
                line.visible = True
            else:
                line.visible = False


class FitTable(Component):
    name = 'FitTable'

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

        self.peak_area = None
        self.peak_height = None

        self.layout = None

    def create(self):
        data = dict(coeff=[], values=[])
        self.cdsource = ColumnDataSource(data)

        columns = [TableColumn(field="coeff", title="Coeff"),
                   TableColumn(field="values", title="Values",
                               formatter=NumberFormatter(format='0.000a'))]
        data_table = DataTable(source=self.cdsource, columns=columns,
                               width=400, height=280)

        self.layout = layout([
            [data_table]
        ])

    def update(self, fitter):
        coeff_dict = fitter.coeff
        cdsource_d = dict(coeff=list(coeff_dict.keys()),
                          values=list(coeff_dict.values()))
        self.cdsource.data = cdsource_d


class BokehSPE(Tool):
    name = "BokehSPE"
    description = "Interactively explore the steps in obtaining and fitting " \
                  "SPE spectrum"

    aliases = Dict(dict(r='EventFileReaderFactory.reader',
                        f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        brightness='FitterWidget.brightness',
                        pe='BokehSPE.adc2pe_path'
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    FitterWidget
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._event = None
        self._event_index = None
        self._event_id = None
        self._active_pixel = 0

        self.w_event_index = None
        self.w_hoa = None
        self.w_fitspectrum = None
        self.w_fitcamera = None
        self.layout = None

        self.reader = None
        self.r1 = None
        self.dl0 = None
        self.gain = None
        self.area = None
        self.height = None

        self.n_events = None
        self.n_pixels = None
        self.n_samples = None

        self.cleaner = None
        self.extractor = None

        self.neighbours2d = None
        self.stage_names = None

        self.p_camera_area = None
        self.p_camera_gain = None
        self.p_fitter = None
        self.p_stage_viewer = None
        self.p_fit_viewer = None
        self.p_fit_table = None

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
        self.p_fitter = FitterWidget(**kwargs)
        self.p_stage_viewer = StageViewer(**kwargs)
        self.p_fit_viewer = FitViewer(**kwargs)
        self.p_fit_table = FitTable(**kwargs)

    def start(self):
        # Prepare storage array
        self.area = np.zeros((self.n_events, self.n_pixels))
        self.height = np.zeros((self.n_events, self.n_pixels))
        self.gain = np.zeros(self.n_pixels)

        source = self.reader.read()
        desc = "Looping through file"
        for event in tqdm(source, total=self.n_events, desc=desc):
            index = event.count

            self.r1.calibrate(event)
            self.dl0.reduce(event)

            telid = list(event.r0.tels_with_data)[0]
            dl0 = np.copy(event.dl0.tel[telid].pe_samples[0])

            # Perform CHECM Waveform Cleaning
            sb_sub_wf, t0 = self.cleaner.apply(dl0)

            # Perform CHECM Charge Extraction
            peak_area, peak_height = self.extractor.extract(sb_sub_wf, t0)

            self.area[index] = peak_area
            self.height[index] = peak_height

        # Setup Plots
        self.p_camera_area.enable_pixel_picker()
        self.p_camera_area.add_colorbar()
        self.p_camera_gain.enable_pixel_picker()
        self.p_camera_gain.add_colorbar()
        self.p_fitter.create()
        self.p_stage_viewer.create(self.neighbours2d, self.stage_names)
        self.p_fit_viewer.create(self.p_fitter.fitter.subfit_labels)
        self.p_fit_table.create()

        # Setup widgets
        self.create_event_index_widget()
        self.event_index = 0
        self.create_hoa_widget()
        self.create_fitspectrum_widget()
        self.create_fitcamera_widget()

        # Get bokeh layouts
        l_camera_area = self.p_camera_area.layout
        l_camera_gain = self.p_camera_gain.layout
        l_fitter = self.p_fitter.layout
        l_stage_viewer = self.p_stage_viewer.layout
        l_fit_viewer = self.p_fit_viewer.layout
        l_fit_table = self.p_fit_table.layout

        # Setup layout
        self.layout = layout([
            [self.w_event_index, self.w_hoa,
             self.w_fitspectrum, self.w_fitcamera],
            [l_camera_area, l_fit_viewer, l_fitter],
            [l_camera_gain, l_fit_table],
            [Div(text="Stage Viewer")],
            [l_stage_viewer],
        ])

    def finish(self):
        curdoc().add_root(self.layout)
        curdoc().title = "Event Viewer"

    def fit_spectrum(self, pix):
        if self.w_hoa.active == 0:
            spectrum = self.area
        else:
            spectrum = self.height

        success = self.p_fitter.fit(spectrum[:, pix])
        if not success:
            self.log.warning("Pixel {} couldn't be fit".format(pix))
        return success

    def fit_camera(self):
        gain = np.ma.zeros(self.n_pixels)
        gain.mask = np.zeros(gain.shape, dtype=np.bool)

        desc = "Extracting gain of pixels"
        for pix in trange(self.n_pixels, desc=desc):
            if not self.fit_spectrum(pix):
                gain.mask[pix] = True
                continue
            gain[pix] = self.p_fitter.fitter.gain

        gain = np.ma.masked_where(np.isnan(gain), gain)
        gain.mask[1664:1728] = True

        self.p_camera_gain.image = gain

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

            self.fit_spectrum(val)

            self.p_camera_area.active_pixel = val
            self.p_camera_gain.active_pixel = val
            self.p_stage_viewer.active_pixel = val

            self.p_fit_viewer.update(self.p_fitter.fitter)
            self.p_fit_table.update(self.p_fitter.fitter)

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

    def create_hoa_widget(self):
        self.w_hoa = RadioGroup(labels=['area', 'height'], active=0)
        self.w_hoa.on_click(self.on_hoa_widget_select)

    def on_hoa_widget_select(self, active):
        self.fit_spectrum(self.active_pixel)
        self.p_fit_viewer.update(self.p_fitter.fitter)
        self.p_fit_table.update(self.p_fitter.fitter)

    def create_fitspectrum_widget(self):
        self.w_fitspectrum = Button(label='Fit Spectrum')
        self.w_fitspectrum.on_click(self.on_fitspectrum_widget_select)

    def on_fitspectrum_widget_select(self):
        t = time()
        self.fit_spectrum(self.active_pixel)
        self.log.info("Fit took {} seconds".format(time()-t))
        self.p_fit_viewer.update(self.p_fitter.fitter)
        self.p_fit_table.update(self.p_fitter.fitter)

    def create_fitcamera_widget(self):
        self.w_fitcamera = Button(label='Fit Camera')
        self.w_fitcamera.on_click(self.on_fitcamera_widget_select)

    def on_fitcamera_widget_select(self):
        self.fit_camera()
        self.fit_spectrum(self.active_pixel)
        self.p_fit_viewer.update(self.p_fitter.fitter)
        self.p_fit_table.update(self.p_fitter.fitter)

exe = BokehSPE()
exe.run()
