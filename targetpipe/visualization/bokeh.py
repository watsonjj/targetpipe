import numpy as np
from bokeh.models import ColumnDataSource, TapTool, palettes, Span, ColorBar, \
    LogColorMapper, ColorMapper, LinearColorMapper
from bokeh.plotting import figure
from bokeh.layouts import row
from targetpipe.utils.plotting import intensity_to_hex
from matplotlib.cm import viridis

PLOTARGS = dict(tools="", toolbar_location=None, outline_line_color='#595959')


class CameraDisplay:
    def __init__(self, geometry=None, image=None, fig=None):
        self._geom = None
        self._image = None
        self._colors = None
        self._image_min = None
        self._image_max = None
        self._fig = None

        self._n_pixels = None
        self._pix_sizes = np.ones(1)
        self._pix_areas = np.ones(1)
        self._pix_x = np.zeros(1)
        self._pix_y = np.zeros(1)

        self.cm = None
        self.cb = None

        cdsource_d = dict(image=[],
                          x=[], y=[],
                          width=[], height=[],
                          outline_color=[], outline_alpha=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self._active_pixels = []
        self.active_index = 0
        self.active_colors = []
        self.automatic_index_increment = False

        self.geom = geometry
        self.image = image
        self.fig = fig

        # TODO: add colorbar to layout
        # self.add_colorbar()
        self.layout = self.fig

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        if val is None:
            val = figure(plot_width=440, plot_height=400, **PLOTARGS)
        val.axis.visible = False
        val.grid.grid_line_color = None
        self._fig = val

        self._draw_camera()

    @property
    def geom(self):
        return self._geom

    @geom.setter
    def geom(self, val):
        self._geom = val

        if val is not None:
            self._pix_areas = val.pix_area.value
            self._pix_sizes = np.sqrt(self._pix_areas)
            self._pix_x = val.pix_x.value
            self._pix_y = val.pix_y.value

        self._n_pixels = self._pix_x.size
        if self._n_pixels == len(self.cdsource.data['x']):
            self.cdsource.data['x'] = self._pix_x
            self.cdsource.data['y'] = self._pix_y
            self.cdsource.data['width'] = self._pix_sizes
            self.cdsource.data['height'] = self._pix_sizes
        else:
            image = np.empty(self._pix_x.shape)
            color = self.cdsource.data['outline_color']
            alpha = self.cdsource.data['outline_alpha']
            cdsource_d = dict(image=image,
                              x=self._pix_x, y=self._pix_y,
                              width=self._pix_sizes, height=self._pix_sizes,
                              outline_color=color, outline_alpha=alpha
                              )
            self.cdsource.data = cdsource_d

        self.active_pixels = [0] * len(self.active_pixels)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        if val is None:
            val = np.zeros(self._n_pixels)

        nonoutliers = val[abs(val - val.mean()) <= 2 * val.std()]
        image_min = nonoutliers.min()
        image_max = nonoutliers.max()
        if image_max == image_min:
            image_min -= 1
            image_max += 1
        colors = intensity_to_hex(val, image_min, image_max)

        self._image = val
        self._colors = colors
        self.image_min = image_min
        self.image_max = image_max

        if len(colors) == self._n_pixels:
            self.cdsource.data['image'] = colors
        else:
            raise ValueError("Image has a different size {} than the current "
                             "CameraGeometry n_pixels {}"
                             .format(colors.size, self._n_pixels))

    @property
    def image_min(self):
        return self._image_min

    @image_min.setter
    def image_min(self, val):
        self._image_min = val
        if self.cb:
            self.cm.low = np.asscalar(val)

    @property
    def image_max(self):
        return self._image_max

    @image_max.setter
    def image_max(self, val):
        self._image_max = val
        if self.cb:
            self.cm.high = np.asscalar(val)

    @property
    def active_pixels(self):
        return self._active_pixels

    @active_pixels.setter
    def active_pixels(self, listval):
        self._active_pixels = listval

        palette = palettes.Set1[9]
        self.active_colors = [palette[i % (len(palette))]
                              for i in range(len(listval))]
        self.highlight_pixels()

    def _draw_camera(self):
        r = self.fig.rect('x', 'y', color='image',
                          width='width', height='height',
                          line_color='outline_color',
                          line_alpha='outline_alpha',
                          line_width=1,
                          nonselection_fill_color='image',
                          nonselection_fill_alpha=1,
                          nonselection_line_color='outline_color',
                          nonselection_line_alpha='outline_alpha',
                          source=self.cdsource)

    def enable_pixel_picker(self, n_active):
        self.active_pixels = [0] * n_active
        self.fig.add_tools(TapTool())

        def source_change_response(attr, old, new):
            val = self.cdsource.selected['1d']['indices']
            if val:
                pix = val[0]
                ai = self.active_index
                self.active_pixels[ai] = pix

                self.highlight_pixels()
                self._on_pixel_click(pix)

                if self.automatic_index_increment:
                    self.active_index = (ai + 1) % len(self.active_pixels)

        self.cdsource.on_change('selected', source_change_response)

    def _on_pixel_click(self, pix_id):
        print("Clicked pixel_id: {}".format(pix_id))
        print("Active Pixels: {}".format(self.active_pixels))

    def highlight_pixels(self):
        alpha = [0] * self._n_pixels
        color = ['black'] * self._n_pixels
        for i, pix in enumerate(self.active_pixels):
            alpha[pix] = 1
            color[pix] = self.active_colors[i]
        self.cdsource.data['outline_alpha'] = alpha
        self.cdsource.data['outline_color'] = color

    def add_colorbar(self):
        self.cm = LinearColorMapper(palette="Viridis256", low=0, high=100,
                                    low_color='white', high_color='red')
        self.cb = ColorBar(color_mapper=self.cm, #label_standoff=6,
                           border_line_color=None,
                           background_fill_alpha=0,
                           major_label_text_color='green',
                           location=(0, 0))
        self.fig.add_layout(self.cb, 'right')
        self.cm.low = np.asscalar(self.image_min)
        self.cm.high = np.asscalar(self.image_max)


class FastCameraDisplay:
    def __init__(self, x_pix, y_pix, pix_size):
        self._image = None
        n_pix = x_pix.size

        cdsource_d = dict(image=np.empty(n_pix, dtype='<U8'), x=x_pix, y=y_pix)
        self.cdsource = ColumnDataSource(cdsource_d)
        self.fig = figure(plot_width=400, plot_height=400, **PLOTARGS)
        self.fig.grid.grid_line_color = None
        self.fig.rect('x', 'y', color='image', source=self.cdsource,
                      width=pix_size[0], height=pix_size[0])

        self.layout = self.fig

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        # Assume colors are already hexidecimal strings, and no geometry
        # is ever changed
        self.cdsource.data['image'] = val


class WaveformDisplay:
    def __init__(self, waveform=np.zeros(1), fig=None):
        self._waveform = None
        self._fig = None
        self._active_time = 0

        self.span = None

        cdsource_d = dict(t=[], samples=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self.waveform = waveform
        self.fig = fig

        self.layout = self.fig

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, val):
        if val is None:
            val = figure(plot_width=700, plot_height=180, **PLOTARGS)
        self._fig = val

        self._draw_waveform()

    @property
    def waveform(self):
        return self._waveform

    @waveform.setter
    def waveform(self, val):
        if val is None:
            val = np.full(1, np.nan)

        self._waveform = val

        if len(val) == len(self.cdsource.data['t']):
            self.cdsource.data['samples'] = val
        else:
            cdsource_d = dict(t=np.arange(val.size), samples=val)
            self.cdsource.data = cdsource_d

    @property
    def active_time(self):
        return self._active_time

    @active_time.setter
    def active_time(self, val):
        max_t = self.cdsource.data['t'][-1]
        if val is None:
            val = 0
        if val < 0:
            val = 0
        if val > max_t:
            val = max_t
        self.span.set(location=val)
        self._active_time = val

    def _draw_waveform(self):
        self.fig.line(x="t", y="samples", source=self.cdsource, name='line')

    def enable_time_picker(self):
        self.span = Span(location=0, dimension='height',
                         line_color='red', line_dash='dashed')
        self.fig.add_layout(self.span)

        self.fig.add_tools(TapTool())

        def wf_tap_response(attr, old, new):
            time = new[0]['x']
            if new[0]['x'] is not None:
                self.active_time = time
                self._on_waveform_click(time)

        self.fig.tool_events.on_change('geometries', wf_tap_response)

    def _on_waveform_click(self, time):
        print("Clicked time: {}".format(time))
        print("Active time: {}".format(self.active_time))
