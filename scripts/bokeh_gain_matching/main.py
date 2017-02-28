import numpy as np
from os.path import dirname, join
from astropy import units as u
from traitlets import Dict, List, Unicode
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, RadioButtonGroup
from ctapipe.core import Tool, Component
from ctapipe.io import CameraGeometry
from targetpipe.visualization.bokeh import CameraDisplay
from targetpipe.io.pixels import checm_pixel_pos, optical_foclen


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

        self.x = None
        self.y = None
        self.x_fit = None
        self.y_fit = None

        self.fig = None
        self.cdsource = None
        self.cdsource_fit = None

        self.layout = None

    def create(self, x, y, m, c):
        self.x = x
        self.y = y
        _, n_pix = self.y.shape
        self.x_fit = np.linspace(self.x[0], self.x[-1], self.x.size*10)
        self.y_fit = m[None, :] * self.x_fit[:, None] + c[None, :]

        self.x = np.exp(self.x)
        self.x_fit = np.exp(self.x_fit)

        max_ = np.max(self.y_fit, axis=1)
        min_ = np.min(self.y_fit, axis=1)
        x_patch = list(self.x_fit) + list(self.x_fit[::-1])
        y_patch = list(max_) + list(min_[::-1])

        self.fig = figure(plot_width=800, plot_height=400)
        self.fig.patch(x_patch, y_patch, color='red', alpha=0.1, line_alpha=0)
        self.fig.xaxis.axis_label = 'HV'
        self.fig.yaxis.axis_label = 'log(Gain)'

        cdsource_d = dict(x=[], y=[])
        cdsource_d_fit = dict(x=[], y=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)
        self.cdsource_fit = ColumnDataSource(data=cdsource_d_fit)
        self.fig.circle('x', 'y', source=self.cdsource)
        self.fig.line('x', 'y', source=self.cdsource_fit)

        self.active_pixel = 0

        self.layout = self.fig

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            cdsource_d = dict(x=self.x, y=self.y[:, val])
            cdsource_d_fit = dict(x=self.x_fit, y=self.y_fit[:, val])
            self.cdsource.data = cdsource_d
            self.cdsource_fit.data = cdsource_d_fit


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

        self.gain = None
        self.hv = None

        self.n_hv = None
        self.n_pixels = None
        self.n_tmpix = 64
        self.modules = None
        self.tmpix = None
        self.n_tm = None

        self.m_pix = None
        self.c_pix = None
        self.m_tm = None
        self.c_tm = None
        self.m_tm2048 = None
        self.c_tm2048 = None

        self.p_camera_pix = None
        self.p_plotter_pix = None
        self.p_camera_tm = None
        self.p_plotter_tm = None

        self.w_view_radio = None

        self.layout = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        arrays = np.load(self.input_path)
        self.gain = arrays['gain']
        self.hv = arrays['hv']

        self.n_hv, self.n_pixels = self.gain.shape
        assert(self.n_hv == self.hv.size)

        geom = CameraGeometry.guess(*checm_pixel_pos * u.m,
                                    optical_foclen * u.m)
        self.modules = np.arange(self.n_pixels) // self.n_tmpix
        self.tmpix = np.arange(self.n_pixels) % self.n_tmpix
        self.n_tm = np.unique(self.modules).size

        # Init Plots
        self.p_camera_pix = Camera(self, "Gain Matching Pixels", geom)
        self.p_camera_tm = Camera(self, "Gain Matching TMs", geom)
        self.p_plotter_pix = Plotter(**kwargs)
        self.p_plotter_tm = Plotter(**kwargs)

    def start(self):
        # Overcomplicated method instead of just reshaping...
        gain_modules = np.zeros((self.n_hv, self.n_tm, self.n_tmpix))
        hv_r = np.arange(self.n_hv, dtype=np.int)[:, None]
        hv_z = np.zeros(self.n_hv, dtype=np.int)[:, None]
        tm_r = np.arange(self.n_tm, dtype=np.int)[None, :]
        tm_z = np.zeros(self.n_tm, dtype=np.int)[None, :]
        tmpix_r = np.arange(self.n_tmpix, dtype=np.int)[None, :]
        tmpix_z = np.zeros(self.n_tmpix, dtype=np.int)[None, :]
        hv_i = (hv_r + tm_z)[..., None] + tmpix_z
        tm_i = (hv_z + tm_r)[..., None] + tmpix_z
        tmpix_i = (hv_z + tm_z)[..., None] + tmpix_r
        gain_rs = np.reshape(self.gain, (self.n_hv, self.n_tm, self.n_tmpix))
        modules_rs = np.reshape(self.modules, (self.n_tm, self.n_tmpix))
        tmpix_rs = np.reshape(self.tmpix, (self.n_tm, self.n_tmpix))
        tm_j = hv_z[..., None] + modules_rs[None, ...]
        tmpix_j = hv_z[..., None] + tmpix_rs[None, ...]
        gain_modules[hv_i, tm_i, tmpix_i] = gain_rs[hv_i, tm_j, tmpix_j]
        gain_modules_mean = np.mean(gain_modules, axis=2)

        x = np.ma.log(self.hv)
        y = np.ma.log(self.gain)
        y_modules = np.ma.log(gain_modules_mean)

        self.m_pix = np.zeros(self.n_pixels)
        self.c_pix = np.zeros(self.n_pixels)
        self.m_tm = np.zeros(self.n_tm)
        self.c_tm = np.zeros(self.n_tm)
        fit = np.ma.polyfit
        for pix in range(self.n_pixels):
            try:
                self.m_pix[pix], self.c_pix[pix] = fit(x, y[:, pix], 1)
            except TypeError:
                self.log.warning("No points for pixel: {}".format(pix))
        for tm in range(self.n_tm):
            self.m_tm[tm], self.c_tm[tm] = fit(x, y_modules[:, tm], 1)

        self.m_tm2048 = self.m_tm[:, None] * np.ones((self.n_tm, self.n_tmpix))
        self.c_tm2048 = self.c_tm[:, None] * np.ones((self.n_tm, self.n_tmpix))

        # Setup Plots
        self.p_camera_pix.enable_pixel_picker()
        self.p_camera_pix.add_colorbar()
        self.p_camera_tm.enable_pixel_picker()
        self.p_camera_tm.add_colorbar()
        self.p_plotter_pix.create(x, y, self.m_pix, self.c_pix)
        self.p_plotter_tm.create(x, y_modules, self.m_tm, self.c_tm)

        # Setup widgets
        self.create_view_radio_widget()
        self.set_camera_image()
        self.active_pixel = 0

        # Get bokeh layouts
        l_camera_pix = self.p_camera_pix.layout
        l_camera_tm = self.p_camera_tm.layout
        l_plotter_pix = self.p_plotter_pix.layout
        l_plotter_tm = self.p_plotter_tm.layout

        # Setup layout
        self.layout = layout([
            [self.w_view_radio],
            [l_camera_pix, l_plotter_pix],
            [l_camera_tm, l_plotter_tm]
        ])

    def finish(self):
        curdoc().add_root(self.layout)
        curdoc().title = "Gain Vs HV"

        output_dir = dirname(self.input_path)
        output_path = join(output_dir, 'gain_matching_coeff.npz')
        np.savez(output_path, alpha_pix=self.m_pix, C_pix=np.exp(self.c_pix),
                 alpha_tm=self.m_tm, C_tm=np.exp(self.c_tm))
        self.log.info("Numpy array saved to: {}".format(output_path))

    @property
    def active_pixel(self):
        return self._active_pixel

    @active_pixel.setter
    def active_pixel(self, val):
        if not self._active_pixel == val:
            self._active_pixel = val
            self.p_camera_pix.active_pixel = val
            self.p_camera_tm.active_pixel = val
            self.p_plotter_pix.active_pixel = val
            self.p_plotter_pix.fig.title.text = 'Pixel {}'.format(val)
            module = self.modules[val]
            self.p_plotter_tm.active_pixel = module
            self.p_plotter_tm.fig.title.text = 'TM {}'.format(module)

    def set_camera_image(self):
        if self.w_view_radio.active == 0:
            self.p_camera_pix.image = self.m_pix
            self.p_camera_tm.image = self.m_tm2048
            self.p_camera_pix.fig.title.text = 'Gain Matching Pixels (gradient)'
            self.p_camera_tm.fig.title.text = 'Gain Matching TMs (gradient)'
        elif self.w_view_radio.active == 1:
            self.p_camera_pix.image = self.c_pix
            self.p_camera_tm.image = self.c_tm2048
            self.p_camera_pix.fig.title.text = 'Gain Matching Pixels (intercept)'
            self.p_camera_tm.fig.title.text = 'Gain Matching TMs (intercept)'

    def create_view_radio_widget(self):
        self.w_view_radio = RadioButtonGroup(labels=["gradient", "intercept"],
                                             active=0)
        self.w_view_radio.on_click(self.on_view_radio_widget_click)

    def on_view_radio_widget_click(self, active):
        self.set_camera_image()


exe = BokehGainMatching()
exe.run()
