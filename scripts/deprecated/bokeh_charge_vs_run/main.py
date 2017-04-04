import numpy as np
from astropy import units as u
from traitlets import Dict, List, Unicode
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, TapTool
from ctapipe.core import Tool, Component
from ctapipe.io import CameraGeometry
from targetpipe.visualization.bokeh import CameraDisplay
from targetpipe.io.pixels import checm_pixel_pos, optical_foclen, Dead
from targetpipe.plots.boxplot import Boxplot


class Camera(CameraDisplay):
    name = 'Camera'

    def __init__(self, parent, title, geometry=None, image=None):
        self._active_pixel = None
        self.parent = parent
        fig = figure(title=title, plot_width=600, plot_height=400, tools="",
                     toolbar_location=None, outline_line_color='#595959')
        super().__init__(geometry=geometry, image=image, fig=fig)

    def enable_pixel_picker(self, _=None):
        super().enable_pixel_picker(1)

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


class Plotter(Component):
    name = 'Plotter'

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

        self._active_pixel = None
        self._active_run = None

        self.x = None
        self.y = None
        self.y_err = None

        self.fig = None
        self.cdsource = None

        self.span = None

        self.layout = None

    def create(self, x, y, y_err):
        self.x = x
        self.y = y
        self.y_err = y_err
        _, n_pix = self.y.shape

        max_ = np.max(self.y, axis=1)
        min_ = np.min(self.y, axis=1)
        x_patch = list(self.x) + list(self.x[::-1])
        y_patch = list(max_) + list(min_[::-1])

        self.fig = figure(plot_width=800, plot_height=400)
        self.fig.patch(x_patch, y_patch, color='red', alpha=0.1, line_alpha=0)
        self.fig.xaxis.axis_label = 'RunDesc'
        self.fig.yaxis.axis_label = 'Charge'

        cdsource_d = dict(x=[], y=[], top=[], bottom=[], left=[], right=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)
        self.fig.circle('x', 'y', source=self.cdsource)

        # Errorbars
        self.fig.segment(x0='x', y0='bottom', x1='x', y1='top',
                         source=self.cdsource, line_width=1.5, color='black')
        self.fig.segment(x0='left', y0='top', x1='right', y1='top',
                         source=self.cdsource, line_width=1.5, color='black')
        self.fig.segment(x0='left', y0='bottom', x1='right', y1='bottom',
                         source=self.cdsource, line_width=1.5, color='black')

        self.active_pixel = 0

        self.layout = self.fig

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val

            top = (self.y + self.y_err)[:, val]
            bottom = (self.y - self.y_err)[:, val]
            left = self.x - 0.3
            right = self.x + 0.3

            cdsource_d = dict(x=self.x, y=self.y[:, val],
                              top=top, bottom=bottom, left=left, right=right)
            self.cdsource.data = cdsource_d

    @property
    def active_run(self):
        return self._active_run

    @active_run.setter
    def active_run(self, val):
        self._active_run = val
        self.span.location = self.x[val]

    def enable_run_picker(self):
        self.span = Span(location=0, dimension='height',
                         line_color='red', line_dash='dashed', line_alpha=0.2)
        self.fig.add_layout(self.span)

        self.fig.add_tools(TapTool())

        def tap_response(attr, old, new):
            x = new[0]['x']
            if x is not None:
                run_i = np.argmin(np.abs(x - self.x))
                self.active_run = run_i
                self._on_click(x)

        self.fig.tool_events.on_change('geometries', tap_response)

    def _on_click(self, x):
        print("Clicked x: {}".format(x))
        print("Active run: {}".format(self.active_run))
        self.parent.active_run = self.active_run


class BoxPlotter(Component):
    name = 'BoxPlotter'

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

        self._active_run = None

        self.x = None

        self.fig = None
        self.bp = None

        self.span = None

        self.layout = None

    def create(self):
        self.fig = figure(plot_width=800, plot_height=400)
        self.fig.xaxis.axis_label = 'RunDesc'
        self.fig.yaxis.axis_label = 'Charge'

        self.bp = Boxplot(self.fig)

        self.layout = self.fig

    def update(self, x, y_data):
        self.x = x
        self.bp.update(x, y_data)

    @property
    def active_run(self):
        return self._active_run

    @active_run.setter
    def active_run(self, val):
        self._active_run = val
        self.span.location = self.x[val]

    def enable_run_picker(self):
        self.span = Span(location=0, dimension='height',
                         line_color='red', line_dash='dashed',
                         line_alpha=0.2)
        self.fig.add_layout(self.span)

        self.fig.add_tools(TapTool())

        def tap_response(attr, old, new):
            x = new[0]['x']
            if x is not None:
                run_i = np.argmin(np.abs(x - self.x))
                self.active_run = run_i
                self._on_click(x)

        self.fig.tool_events.on_change('geometries', tap_response)

    def _on_click(self, x):
        print("Clicked x: {}".format(x))
        print("Active run: {}".format(self.active_run))
        self.parent.active_run = self.active_run


class BokehGainMatching(Tool):
    name = "BokehGainMatching"
    description = "Interactively explore the steps in obtaining charge vs hv"

    input_path = Unicode('', help='Path to the numpy array containing the '
                                  'gain and hv').tag(config=True)

    aliases = Dict(dict(f='BokehGainMatching.input_path'
                        ))

    classes = List([

    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._active_pixel = None
        self._active_run = None

        self.dead = Dead()
        self.charge = None
        self.charge_error = None
        self.rundesc = None
        self.charge_tm = None
        self.charge_error_tm = None
        self.mean_tm2048 = None
        self.tmpixspread_tm2048 = None

        self.n_run = None
        self.n_pixels = None
        self.n_tmpix = 64
        self.modules = None
        self.tmpix = None
        self.n_tm = None

        self.p_c_pix = None
        self.p_p_pix = None
        self.p_c_tm = None
        self.p_p_tm = None
        self.p_c_tmpixspread = None
        self.p_b_tmpixspread = None
        self.p_b_tmspread = None
        self.p_b_pixspread = None

        self.p_c_pix_title = 'Charge Across Pixels, Run: {}'
        self.p_c_tm_title = 'Mean Charge Across TMs, Run: {}'
        self.p_c_tmpixspread_title = 'Median Charge Across TMs, Run: {}'
        self.p_p_pix_title = 'Charge vs Runs, Error: fit stddev, Pixel: {}'
        self.p_p_tm_title = 'Charge vs Runs, Error: combined pixel, TM: {}'
        self.p_b_tmpixspread_title = 'Charge Spread vs Runs, TM: {}'

        self.layout = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        arrays = np.load(self.input_path)
        self.charge = self.dead.mask2d(arrays['charge'])
        self.charge_error = self.dead.mask2d(arrays['charge_error'])
        self.rundesc = arrays['rundesc']

        self.n_run, self.n_pixels = self.charge.shape
        assert(self.n_run == self.rundesc.size)

        geom = CameraGeometry.guess(*checm_pixel_pos * u.m,
                                    optical_foclen * u.m)
        self.modules = np.arange(self.n_pixels) // self.n_tmpix
        self.tmpix = np.arange(self.n_pixels) % self.n_tmpix
        self.n_tm = np.unique(self.modules).size

        # Init Plots
        self.p_c_pix = Camera(self, "", geom)
        self.p_c_tm = Camera(self, "", geom)
        self.p_c_tmpixspread = Camera(self, "", geom)
        self.p_p_pix = Plotter(**kwargs)
        self.p_p_tm = Plotter(**kwargs)
        self.p_b_tmpixspread = BoxPlotter(**kwargs)
        self.p_b_tmspread = BoxPlotter(**kwargs)
        self.p_b_pixspread = BoxPlotter(**kwargs)

    def start(self):
        shape_tm = (self.n_run, self.n_tm, self.n_tmpix)
        shape_4d = (self.n_run, self.n_tm, self.n_tmpix, self.n_tmpix)
        shape_pix = (self.n_run, self.n_pixels, self.n_tmpix)

        self.charge_tm = np.reshape(self.charge, shape_tm)
        self.charge_error_tm = np.reshape(self.charge_error, shape_tm)
        charge_tm_mean = np.mean(self.charge_tm, axis=2)
        charge_error_tm_mean = np.sqrt(np.sum(self.charge_error_tm**2, axis=2))
        self.mean_tm2048 = charge_tm_mean[..., None] * np.ones(shape_tm)
        tm_spread = self.charge_tm[:, :, None, :] * np.ones(shape_4d)
        self.tmpixspread_tm2048 = np.reshape(tm_spread, shape_pix)

        # Setup Plots
        self.p_c_pix.enable_pixel_picker()
        self.p_c_pix.add_colorbar()
        self.p_c_tm.enable_pixel_picker()
        self.p_c_tm.add_colorbar()
        self.p_c_tmpixspread.enable_pixel_picker()
        self.p_c_tmpixspread.add_colorbar()
        self.p_p_pix.create(self.rundesc, self.charge, self.charge_error)
        self.p_p_tm.create(self.rundesc, charge_tm_mean, charge_error_tm_mean)
        self.p_b_tmpixspread.create()
        self.p_b_tmspread.create()
        self.p_b_pixspread.create()

        self.p_b_tmspread.fig.title.text = 'Mean TM Charge Spread vs Runs'
        self.p_b_pixspread.fig.title.text = 'Pixel Spread vs Runs'

        self.p_b_tmspread.update(self.rundesc, charge_tm_mean)
        self.p_b_pixspread.update(self.rundesc, self.charge)

        self.p_p_pix.enable_run_picker()
        self.p_p_tm.enable_run_picker()
        self.p_b_tmpixspread.enable_run_picker()
        self.p_b_tmspread.enable_run_picker()
        self.p_b_pixspread.enable_run_picker()

        # Setup widgets
        self.active_pixel = 0
        self.active_run = 0

        # Get bokeh layouts
        l_camera_pix = self.p_c_pix.layout
        l_camera_tm = self.p_c_tm.layout
        l_camera_tmpixspread = self.p_c_tmpixspread.layout
        l_plotter_pix = self.p_p_pix.layout
        l_plotter_tm = self.p_p_tm.layout
        l_boxplotter_tmpixspread = self.p_b_tmpixspread.layout
        l_boxplotter_tmspread = self.p_b_tmspread.layout
        l_boxplotter_pixspread = self.p_b_pixspread.layout

        # Setup layout
        self.layout = layout([
            [l_camera_pix, l_plotter_pix],
            [l_camera_tm, l_plotter_tm],
            [l_camera_tmpixspread, l_boxplotter_tmpixspread],
            [l_boxplotter_tmspread, l_boxplotter_pixspread]
        ])

    def finish(self):
        curdoc().add_root(self.layout)
        curdoc().title = "Charge Vs Run"

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            self.p_c_pix.active_pixel = val
            self.p_c_tm.active_pixel = val
            self.p_c_tmpixspread.active_pixel = val
            self.p_p_pix.active_pixel = val
            self.p_p_pix.fig.title.text = self.p_p_pix_title.format(val)
            module = self.modules[val]
            self.p_p_tm.active_pixel = module
            self.p_p_tm.fig.title.text = self.p_p_tm_title.format(module)
            self.p_b_tmpixspread.update(self.rundesc, self.tmpixspread_tm2048[:, val])
            t = self.p_b_tmpixspread_title
            self.p_b_tmpixspread.fig.title.text = t.format(module)

    @property
    def active_run(self):
        return self._active_run

    @active_run.setter
    def active_run(self, val):
        if not self._active_run == val:
            self._active_run = val
            self.p_p_pix.active_run = val
            self.p_p_tm.active_run = val
            self.p_b_tmpixspread.active_run = val
            self.p_b_tmspread.active_run = val
            self.p_b_pixspread.active_run = val
            self.set_camera_image()
            self.p_c_pix.fig.title.text = self.p_c_pix_title.format(val)
            self.p_c_tm.fig.title.text = self.p_c_tm_title.format(val)
            t = self.p_c_tmpixspread_title
            self.p_c_tmpixspread.fig.title.text = t.format(val)

    def set_camera_image(self):
        r = self.active_run
        self.p_c_pix.image = self.charge[r]
        self.p_c_tm.image = self.mean_tm2048[r]
        self.p_c_tmpixspread.image = np.median(self.tmpixspread_tm2048[r], axis=1)

exe = BokehGainMatching()
exe.run()
