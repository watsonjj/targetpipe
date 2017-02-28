import numpy as np
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models.widgets import Select, PreText
from bokeh.models import HoverTool, Range1d, ColumnDataSource
from traitlets import Dict, List, Unicode
from ctapipe.core import Tool, Component
from ctapipe.io import CameraGeometry
from targetpipe.io.eventfilereader import TargetioFileReader
from targetpipe.calib.camera.pedestal import PedestalSubtractor
from targetpipe.visualization.bokeh import CameraDisplay


class PedCellSpread(Component):
    name = 'PedCellSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of Pedestal across pixels for each
        cell

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

        self.layout = None

    def create(self, ped):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(ped, axis=(0, 1))
        stddev = np.std(ped, axis=(0, 1))
        min_ = np.min(ped, axis=(0, 1))
        max_ = np.max(ped, axis=(0, 1))
        x = np.arange(ped.shape[2])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Pedestal Spread Vs Cell ID"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("min", "@min"),
                                          ("max", "@max"),
                                          ], renderers=[c]))

        ped_min = ped.min()
        ped_max = ped.max()
        diff = ped_max - ped_min
        y_min = ped_min - diff * 0.1
        y_max = ped_max + diff * 0.1
        x_min = x.min() - 100
        x_max = x.max() + 100
        fig.y_range = Range1d(y_min, y_max,
                              bounds=(y_min - 100, y_max + 100))
        fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        fig.xaxis.axis_label = 'Cell'
        fig.yaxis.axis_label = 'Pedestal'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class PedPixelSpread(Component):
    name = 'PedPixelSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of Pedestal across cells for each
        pixel

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

        self.layout = None

    def create(self, ped):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(ped, axis=2).flatten()
        stddev = np.std(ped, axis=2).flatten()
        min_ = np.min(ped, axis=2).flatten()
        max_ = np.max(ped, axis=2).flatten()
        x = np.arange(ped.shape[0] * ped.shape[1])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Pedestal Spread Vs Pixel ID"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("min", "@min"),
                                          ("max", "@max"),
                                          ], renderers=[c]))

        ped_min = ped.min()
        ped_max = ped.max()
        diff = ped_max - ped_min
        y_min = ped_min - diff * 0.1
        y_max = ped_max + diff * 0.1
        x_min = x.min() - 100
        x_max = x.max() + 100
        fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        fig.xaxis.axis_label = 'Pixel (TM * N_TMPIX + TMPIX)'
        fig.yaxis.axis_label = 'Pedestal'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class PedSelector(Component):
    name = 'PedSelector'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the ped of a selected tm, tmpix

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

        self.layout = None

        self._ped = None
        self._stddev = None

        self.n_tmpix = None
        self._tm = 0
        self._tmpix = 0
        self._pixel = 0

        cdsource_d = dict(x=[], y=[], y_err=[],
                          top=[], bottom=[], left=[], right=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self.tm_select = None
        self.tmpix_select = None
        self.pixel_select = None
        self.title = None

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self, val):
        self._tm = val
        self._pixel = val * self.n_tmpix + self.tmpix
        self.pixel_select.value = str(self.pixel)
        self._update_cdsource()

    @property
    def tmpix(self):
        return self._tmpix

    @tmpix.setter
    def tmpix(self, val):
        self._tmpix = val
        self._pixel = self.tm * self.n_tmpix + val
        self.pixel_select.value = str(self.pixel)
        self._update_cdsource()

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, val):
        self._pixel = val
        self._tm = val // self.n_tmpix
        self._tmpix = val % self.n_tmpix
        self.tm_select.value = str(self.tm)
        self.tmpix_select.value = str(self.tmpix)
        self._update_cdsource()

    def _update_cdsource(self):
        x = np.arange(self._ped.shape[2])
        y = self._ped[self.tm, self.tmpix]

        y_err = np.zeros(y.shape)
        if self._stddev is not None:
            y_err = self._stddev[self.tm * self.n_tmpix + self.tmpix]
        top = y + y_err
        bottom = y - y_err
        left = x - 0.3
        right = x + 0.3

        cdsource_d = dict(x=x, y=y, y_err=y_err,
                          top=top, bottom=bottom, left=left, right=right)
        self.cdsource.data = cdsource_d

        if self.title:
            self.title.text = "TF Selector  TM: {}, TMPIX: {}" \
                .format(self.tm, self.tmpix)

    def create(self, ped, stddev=None):
        self.log.info("Creating {}".format(self.name))
        self._ped = ped
        self._stddev = stddev

        n_tm = ped.shape[0]
        self.n_tmpix = ped.shape[1]
        n_pixels = n_tm * self.n_tmpix

        self._update_cdsource()

        fig = self._get_figure()

        # Widgets
        tm_str = ['{:.0f}'.format(x) for x in range(n_tm)]
        tmpix_str = ['{:.0f}'.format(x) for x in range(self.n_tmpix)]
        pixel_str = ['{:.0f}'.format(x) for x in range(n_pixels)]

        self.tm_select = Select(title="TM:", value='0', options=tm_str)
        self.tmpix_select = Select(title="TMPIX:", value='0',
                                   options=tmpix_str)
        self.pixel_select = Select(title="PIXEL:", value='0',
                                   options=pixel_str)

        text = PreText(text="""
        Changing of TM and TMPIX only works when plot_pedestal
        is ran from inside a Bokeh server.
        e.g.
        bokeh serve --show plot_pedestal.py --args -P pedestal.fits
        """, width=500, height=100)

        self.tm_select.on_change('value', self._on_tm_select)
        self.tmpix_select.on_change('value', self._on_tmpix_select)
        self.pixel_select.on_change('value', self._on_pixel_select)

        self.layout = layout([
            [self.tm_select, self.tmpix_select, self.pixel_select, text],
            [fig]
        ])

    def _get_figure(self):
        title = "TF Selector  TM: {}, TMPIX: {}".format(self.tm, self.tmpix)
        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        self.title = fig.title
        c = fig.circle(source=self.cdsource, x='x', y='y', hover_color="red")
        fig.segment(source=self.cdsource,
                    x0='x', y0='bottom', x1='x', y1='top',
                    line_width=1.5, color='black',
                    visible=self._stddev is not None)
        fig.segment(source=self.cdsource,
                    x0='left', y0='top', x1='right', y1='top',
                    line_width=1.5, color='black',
                    visible=self._stddev is not None)
        fig.segment(source=self.cdsource,
                    x0='left', y0='bottom', x1='right', y1='bottom',
                    line_width=1.5, color='black',
                    visible=self._stddev is not None)
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)"),
                                          ("stddev", "@y_err")],
                                renderers=[c]))

        n_cells = self._ped.shape[2]
        x_min = -100
        x_max = n_cells + 100

        fig.x_range = Range1d(x_min, x_max, bounds=(x_min - 400, x_max + 400))
        fig.xaxis.axis_label = 'Cell'
        fig.yaxis.axis_label = 'Pedestal'

        return fig

    def _on_tm_select(self, attr, old, new):
        self.tm = int(self.tm_select.value)

    def _on_tmpix_select(self, attr, old, new):
        self.tmpix = int(self.tmpix_select.value)

    def _on_pixel_select(self, attr, old, new):
        self.pixel = int(self.pixel_select.value)


class StddevCellSpread(Component):
    name = 'StddevCellSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of Stddev across pixels for each
        cell

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

        self.layout = None

    def create(self, in_stddev):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(in_stddev, axis=0)
        stddev = np.std(in_stddev, axis=0)
        min_ = np.min(in_stddev, axis=0)
        max_ = np.max(in_stddev, axis=0)
        x = np.arange(in_stddev.shape[1])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Input Standard Deviation Spread Vs Cell ID"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("min", "@min"),
                                          ("max", "@max"),
                                          ], renderers=[c]))

        ped_min = in_stddev.min()
        ped_max = in_stddev.max()
        diff = ped_max - ped_min
        y_min = ped_min - diff * 0.1
        y_max = ped_max + diff * 0.1
        x_min = x.min() - 100
        x_max = x.max() + 100
        fig.y_range = Range1d(y_min, y_max,
                              bounds=(y_min - 100, y_max + 100))
        fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        fig.xaxis.axis_label = 'Cell'
        fig.yaxis.axis_label = 'Standard Deviation'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class StddevPixelSpread(Component):
    name = 'StddevPixelSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of Stddev across cells for each
        pixel

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

        self.layout = None

    def create(self, in_stddev):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(in_stddev, axis=1)
        stddev = np.std(in_stddev, axis=1)
        min_ = np.min(in_stddev, axis=1)
        max_ = np.max(in_stddev, axis=1)
        x = np.arange(in_stddev.shape[0])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Input Standard Deviation Spread Vs Pixel ID"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("min", "@min"),
                                          ("max", "@max"),
                                          ], renderers=[c]))

        ped_min = in_stddev.min()
        ped_max = in_stddev.max()
        diff = ped_max - ped_min
        y_min = ped_min - diff * 0.1
        y_max = ped_max + diff * 0.1
        x_min = x.min() - 100
        x_max = x.max() + 100
        fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        fig.xaxis.axis_label = 'Pixel (TM * N_TMPIX + TMPIX)'
        fig.yaxis.axis_label = 'Standard Deviation'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class ResidualPixelSpread(Component):
    name = 'ResidualPixelSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of the residuals for each pixel

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

        self.layout = None

    def create(self, residual):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(residual, axis=(0, 2))
        stddev = np.std(residual, axis=(0, 2))
        min_ = np.min(residual, axis=(0, 2))
        max_ = np.max(residual, axis=(0, 2))
        x = np.arange(residual.shape[1])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Residual Spread Vs Pixel ID"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("min", "@min"),
                                          ("max", "@max"),
                                          ], renderers=[c]))

        residual_min = residual.min()
        residual_max = residual.max()
        diff = residual_max - residual_min
        y_min = residual_min - diff * 0.1
        y_max = residual_max + diff * 0.1
        x_min = x.min() - 100
        x_max = x.max() + 100
        fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        fig.xaxis.axis_label = 'Pixel (TM * N_TMPIX + TMPIX)'
        fig.yaxis.axis_label = 'Pedestal Subtracted ADC'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class ResidualCameraSpread(Component):
    name = 'ResidualCameraSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of the residuals across the camera

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

        self.layout = None

    def create(self, residual, pixel_pos, foclen):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(residual, axis=(0, 2))
        stddev = np.std(residual, axis=(0, 2))
        min_ = np.min(residual, axis=(0, 2))
        max_ = np.max(residual, axis=(0, 2))
        x = np.arange(residual.shape[1])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Residual Spread Vs Pixel ID"

        geom = CameraGeometry.guess(*pixel_pos, foclen)
        camera_mean = CameraDisplay(geometry=geom, image=mean)
        camera_mean.add_colorbar()
        camera_mean_fig = camera_mean.fig
        camera_mean_fig.title.text = "Residual Means"
        camera_stddev = CameraDisplay(geometry=geom, image=stddev)
        camera_stddev.add_colorbar()
        camera_stddev_fig = camera_stddev.fig
        camera_stddev_fig.title.text = "Residual Standard Deviations"

        self.layout = layout([[camera_mean_fig, camera_stddev_fig]])


class ResidualEventSpread(Component):
    name = 'ResidualEventSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of the residuals for each event

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

        self.layout = None

    def create(self, residual):
        self.log.info("Creating {}".format(self.name))
        mean = np.mean(residual, axis=(1, 2))
        stddev = np.std(residual, axis=(1, 2))
        min_ = np.min(residual, axis=(1, 2))
        max_ = np.max(residual, axis=(1, 2))
        x = np.arange(residual.shape[0])

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "Residual Spread Vs Event"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("min", "@min"),
                                          ("max", "@max"),
                                          ], renderers=[c]))

        residual_min = residual.min()
        residual_max = residual.max()
        diff = residual_max - residual_min
        y_min = residual_min - diff * 0.1
        y_max = residual_max + diff * 0.1
        x_min = x.min() - 100
        x_max = x.max() + 100
        fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        fig.x_range = Range1d(x_min, x_max, bounds=(x_min, x_max))
        fig.xaxis.axis_label = 'Event'
        fig.yaxis.axis_label = 'Pedestal Subtracted ADC'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='red')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='red')

        # Errorbars
        top = mean + stddev
        bottom = mean - stddev
        left = x - 0.3
        right = x + 0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class ResidualSelector(Component):
    name = 'ResidualSelector'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the residuals of a selected tm, tmpix

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

        self.layout = None

        self._residuals = None
        self._cells = None
        self._n_cells = None

        self.n_tmpix = None
        self._tm = 0
        self._tmpix = 0
        self._pixel = 0

        cdsource_d = dict(x=[], y=[], y_err=[],
                          top=[], bottom=[], left=[], right=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self.tm_select = None
        self.tmpix_select = None
        self.pixel_select = None
        self.title = None

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self, val):
        self._tm = val
        self._pixel = val * self.n_tmpix + self.tmpix
        self.pixel_select.value = str(self.pixel)
        self._update_cdsource()

    @property
    def tmpix(self):
        return self._tmpix

    @tmpix.setter
    def tmpix(self, val):
        self._tmpix = val
        self._pixel = self.tm * self.n_tmpix + val
        self.pixel_select.value = str(self.pixel)
        self._update_cdsource()

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, val):
        self._pixel = val
        self._tm = val // self.n_tmpix
        self._tmpix = val % self.n_tmpix
        self.tm_select.value = str(self.tm)
        self.tmpix_select.value = str(self.tmpix)
        self._update_cdsource()

    def _update_cdsource(self):
        pixel_cells = self._cells[:, self.pixel]
        pixel_res = self._residuals[:, self.pixel]
        n_events, n_samples = pixel_cells.shape
        event_vs_cells = np.zeros((n_events, self._n_cells))
        event_index = np.arange(n_events)[:, None] + np.zeros_like(pixel_cells)
        event_vs_cells[event_index, pixel_cells] = pixel_res
        sp = np.ma.masked_where(event_vs_cells == 0, event_vs_cells)

        x = np.arange(self._n_cells)
        y = np.mean(sp, axis=0)
        y_err = np.std(sp, axis=0)

        top = y + y_err
        bottom = y - y_err
        left = x - 0.3
        right = x + 0.3

        cdsource_d = dict(x=x, y=y, y_err=y_err,
                          top=top, bottom=bottom, left=left, right=right)
        self.cdsource.data = cdsource_d

        if self.title:
            self.title.text = "Residuals  TM: {}, TMPIX: {}" \
                .format(self.tm, self.tmpix)

    def create(self, residuals, cells, n_modules, n_cells):
        self.log.info("Creating {}".format(self.name))
        self._residuals = residuals
        self._cells = cells.astype(np.int)
        self._n_cells = n_cells

        n_tm = n_modules
        n_pixels = residuals.shape[1]
        self.n_tmpix = n_pixels // n_modules

        self._update_cdsource()

        fig = self._get_figure()

        # Widgets
        tm_str = ['{:.0f}'.format(x) for x in range(n_tm)]
        tmpix_str = ['{:.0f}'.format(x) for x in range(self.n_tmpix)]
        pixel_str = ['{:.0f}'.format(x) for x in range(n_pixels)]

        self.tm_select = Select(title="TM:", value='0', options=tm_str)
        self.tmpix_select = Select(title="TMPIX:", value='0',
                                   options=tmpix_str)
        self.pixel_select = Select(title="PIXEL:", value='0',
                                   options=pixel_str)

        self.tm_select.on_change('value', self._on_tm_select)
        self.tmpix_select.on_change('value', self._on_tmpix_select)
        self.pixel_select.on_change('value', self._on_pixel_select)

        self.layout = layout([
            [self.tm_select, self.tmpix_select, self.pixel_select],
            [fig]
        ])

    def _get_figure(self):
        title = "Residuals  TM: {}, TMPIX: {}".format(self.tm, self.tmpix)
        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        self.title = fig.title
        c = fig.circle(source=self.cdsource, x='x', y='y', hover_color="red")
        fig.segment(source=self.cdsource,
                    x0='x', y0='bottom', x1='x', y1='top',
                    line_width=1.5, color='black')
        fig.segment(source=self.cdsource,
                    x0='left', y0='top', x1='right', y1='top',
                    line_width=1.5, color='black')
        fig.segment(source=self.cdsource,
                    x0='left', y0='bottom', x1='right', y1='bottom',
                    line_width=1.5, color='black')
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)"),
                                          ("stddev", "@y_err")],
                                renderers=[c]))

        x_min = -100
        x_max = self._n_cells + 100

        fig.x_range = Range1d(x_min, x_max, bounds=(x_min - 400, x_max + 400))
        fig.xaxis.axis_label = 'Cell'
        fig.yaxis.axis_label = 'Pedestal Subtracted ADC'

        return fig

    def _on_tm_select(self, attr, old, new):
        self.tm = int(self.tm_select.value)

    def _on_tmpix_select(self, attr, old, new):
        self.tmpix = int(self.tmpix_select.value)

    def _on_pixel_select(self, attr, old, new):
        self.pixel = int(self.pixel_select.value)


class TargetCalibPedestalExplorer(Tool):
    name = "TargetCalibPedestalExplorer"
    description = "Plot the TargetCalib pedestal using bokeh"

    input_path = Unicode(None, allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)
    ped_stddev_path = Unicode(None, allow_none=True,
                              help='Path to the stddev numpy file created by '
                                   'generate_pedestal.py').tag(config=True)

    aliases = Dict(dict(f='TargetCalibPedestalExplorer.input_path',
                        max_events='TargetioFileReader.max_events',
                        P='PedestalSubtractor.pedestal_path',
                        stddev='TargetCalibPedestalExplorer.ped_stddev_path'
                        ))
    classes = List([TargetioFileReader,
                    PedestalSubtractor
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.extractor = None
        self.calibrator = None
        self.plotter = None
        self.ped = None

        self.p_cellspread = None
        self.p_pixelspread = None
        self.p_pedselect = None
        self.p_devcellspread = None
        self.p_devpixelspread = None
        self.p_respixelspread = None
        self.p_rescameraspread = None
        self.p_reseventspread = None
        self.p_resselect = None

        self.layout = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        # Bypass error message when input_path is None
        if self.input_path is not None:
            self.file_reader = TargetioFileReader(**kwargs,
                                                  input_path=self.input_path)

        self.ped = PedestalSubtractor(**kwargs)

        self.p_cellspread = PedCellSpread(**kwargs)
        self.p_pixelspread = PedPixelSpread(**kwargs)
        self.p_pedselect = PedSelector(**kwargs)
        self.p_devcellspread = StddevCellSpread(**kwargs)
        self.p_devpixelspread = StddevPixelSpread(**kwargs)
        self.p_respixelspread = ResidualPixelSpread(**kwargs)
        self.p_rescameraspread = ResidualCameraSpread(**kwargs)
        self.p_reseventspread = ResidualEventSpread(**kwargs)
        self.p_resselect = ResidualSelector(**kwargs)

    def start(self):
        # Get Ped
        ped = self.ped.get_ped()

        # Get Ped Stddev
        ped_stddev = None
        if self.ped_stddev_path:
            self.log.info("Loading pedestal stddev file: {}"
                          .format(self.ped_stddev_path))
            ped_stddev = np.load(self.ped_stddev_path)

        # Get Events
        residuals = None
        cells = None
        n_modules = None
        n_cells = None
        pixel_pos = None
        foclen = None
        if self.file_reader:
            n_events = self.file_reader.num_events
            first_event = self.file_reader.get_event(0)
            first_wf = first_event.r0.tel[0].adc_samples[0]
            n_pix, n_samples = first_wf.shape
            n_modules = first_event.meta['n_modules']
            n_cells = first_event.meta['n_cells']
            pixel_pos = first_event.inst.pixel_pos[0]
            foclen = first_event.inst.optical_foclen[0]

            residuals = np.zeros((n_events, n_pix, n_samples),
                                 dtype=np.float32)
            cells = np.zeros((n_events, n_pix, n_samples))

            source = self.file_reader.read()
            for event in source:
                index = event.count
                self.ped.apply(event, residuals[index])
                fci = event.r0.tel[0].first_cell_ids
                cells[index] = (np.arange(n_samples)[None, :] +
                                fci[:, None]) % n_cells

        # TEMP!!!!!!!! Remove channel 4
        # ped[0, 4] = 0

        # # Mask zero values
        # ped = np.ma.masked_where(ped == 0, ped)
        # mask_3d = ped.mask
        # if mask_3d:
        #     mask_3d = np.zeros(ped.shape, dtype=bool)
        # shape_2d = [mask_3d.shape[0] * mask_3d.shape[1], mask_3d.shape[2]]
        # mask_2d = mask_3d.reshape(shape_2d)
        # if ped_stddev is not None:
        #     ped_stddev = np.ma.masked_array(ped_stddev, mask_2d)

        # Create bokeh figures
        self.p_cellspread.create(ped)
        self.p_pixelspread.create(ped)
        self.p_pedselect.create(ped, ped_stddev)
        if ped_stddev is not None:
            self.p_devcellspread.create(ped_stddev)
            self.p_devpixelspread.create(ped_stddev)
        if residuals is not None:
            self.p_respixelspread.create(residuals)
            self.p_rescameraspread.create(residuals, pixel_pos, foclen)
            self.p_reseventspread.create(residuals)
            self.p_resselect.create(residuals, cells, n_modules, n_cells)

        # Get bokeh layouts
        l_cellspread = self.p_cellspread.layout
        l_pixelspread = self.p_pixelspread.layout
        l_pedselect = self.p_pedselect.layout
        l_devcellspread = self.p_devcellspread.layout
        l_devpixelspread = self.p_devpixelspread.layout
        l_respixelspread = self.p_respixelspread.layout
        l_rescameraspread = self.p_rescameraspread.layout
        l_reseventspread = self.p_reseventspread.layout
        l_resselect = self.p_resselect.layout

        # Get widgets

        # Layout
        layout_list = [
            [l_cellspread],
            [l_pixelspread],
            [l_pedselect]
        ]
        if ped_stddev is not None:
            layout_list.append([l_devcellspread])
            layout_list.append([l_devpixelspread])
        if residuals is not None:
            layout_list.append([l_respixelspread])
            layout_list.append([l_rescameraspread])
            layout_list.append([l_reseventspread])
            layout_list.append([l_resselect])
        self.layout = layout(layout_list, sizing_mode="scale_width")

    def finish(self):
        curdoc().add_root(self.layout)
        curdoc().title = "Pedestal"


exe = TargetCalibPedestalExplorer()
exe.run()
