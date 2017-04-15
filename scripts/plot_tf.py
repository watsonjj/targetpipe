"""
Create diagnostic plots (bokeh) for the TF
"""

import numpy as np
from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import layout
from bokeh.models.widgets import Select
from bokeh.models import HoverTool, ColumnDataSource
from traitlets import Dict, List, Unicode
from ctapipe.core import Tool, Component
from targetpipe.calib.camera.tf import TFApplier
from os.path import join, exists, dirname
from os import makedirs



class TFSpread(Component):
    name = 'TFSpread'

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

        self.layout = None

    def create(self, tf, adc_min, adc_step):
        mean = np.mean(tf, axis=(0, 1, 2))
        stddev = np.std(tf, axis=(0, 1, 2))
        min_ = np.min(tf, axis=(0, 1, 2))
        max_ = np.max(tf, axis=(0, 1, 2))
        x = adc_min + np.arange(tf.shape[3]) * adc_step

        title = "TF Spread"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(x=x, y=mean, hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                                renderers=[c]))
        # fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        # fig.x_range = Range1d(-100, n_cells + 100, bounds=(-500, n_cells + 500))
        fig.xaxis.axis_label = 'ADC'
        fig.yaxis.axis_label = 'VPed'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        # fig.segment(x0=x, y0=bottom, x1=x, y1=top,
        #             line_width=1.5, color='red')
        # fig.segment(x0=left, y0=top, x1=right, y1=top,
        #             line_width=1.5, color='red')
        # fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
        #             line_width=1.5, color='red')

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


class TFSelector(Component):
    name = 'TFSelector'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the TF of a selected tm, tmpix, cell

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

        self._tf = None
        self._adc_min = None
        self._adc_step = None

        self._tm = 0
        self._tmpix = 0
        self._cell = 0

        cdsource_d = dict(x=[], y=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self.tm_select = None
        self.tmpix_select = None
        self.cell_select = None

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self, val):
        self._tm = val
        self._update_cdsource()

    @property
    def tmpix(self):
        return self._tmpix

    @tmpix.setter
    def tmpix(self, val):
        self._tmpix = val
        self._update_cdsource()

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, val):
        self._cell = val
        self._update_cdsource()

    def _update_cdsource(self):
        x = self._adc_min + np.arange(self._tf.shape[3]) * self._adc_step
        y = self._tf[self.tm, self.tmpix, self.cell]
        a = np.where(y > 0)
        cdsource_d = dict(x=x, y=y)
        self.cdsource.data = cdsource_d
        # self.cdsource.trigger('data', None, None)
        # if self.layout:
        #     self.layout.children[1].children[0] = self.get_figure()

    def create(self, tf, adc_min, adc_step):
        self._tf = tf
        self._adc_min = adc_min
        self._adc_step = adc_step

        n_tm = tf.shape[0]
        n_tmpix = tf.shape[1]
        n_cells = tf.shape[2]

        self._update_cdsource()

        fig = self._get_figure()

        # Widgets
        tm_str = ['{:.0f}'.format(x) for x in range(n_tm)]
        tmpix_str = ['{:.0f}'.format(x) for x in range(n_tmpix)]
        cell_str = ['{:.0f}'.format(x) for x in range(n_cells)]

        self.tm_select = Select(title="TM:", value='0', options=tm_str)
        self.tmpix_select = Select(title="TMPIX:", value='0',
                                   options=tmpix_str)
        self.cell_select = Select(title="CELL:", value='0', options=cell_str)

        self.tm_select.on_change('value', self._on_tm_select)
        self.tmpix_select.on_change('value', self._on_tmpix_select)
        self.cell_select.on_change('value', self._on_cell_select)

        self.layout = layout([
            [self.tm_select, self.tmpix_select, self.cell_select],
            [fig]
        ])

    def _get_figure(self):
        title = "TF Selector"
        # tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        tools = "pan, box_zoom, wheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='wheel_zoom', webgl=True)
        c = fig.circle(source=self.cdsource, x='x', y='y', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                                renderers=[c]))
        # fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        # fig.x_range = Range1d(-100, n_cells + 100, bounds=(-500, n_cells + 500))
        fig.xaxis.axis_label = 'ADC'
        fig.yaxis.axis_label = 'VPed'

        return fig

    def _on_tm_select(self, attr, old, new):
        self.tm = int(self.tm_select.value)

    def _on_tmpix_select(self, attr, old, new):
        self.tmpix = int(self.tmpix_select.value)

    def _on_cell_select(self, attr, old, new):
        self.cell = int(self.cell_select.value)


class TFInputSpread(Component):
    name = 'TFSpread'

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

        self.layout = None

    def create(self, tf_input, vped):
        mean = np.mean(tf_input, axis=(0, 1, 2))
        stddev = np.std(tf_input, axis=(0, 1, 2))
        min_ = np.min(tf_input, axis=(0, 1, 2))
        max_ = np.max(tf_input, axis=(0, 1, 2))
        x = vped

        title = "TFInput Spread"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(x=x, y=mean, hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                                renderers=[c]))
        # fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        # fig.x_range = Range1d(-100, n_cells + 100, bounds=(-500, n_cells + 500))
        fig.xaxis.axis_label = 'Vped'
        fig.yaxis.axis_label = 'ADC'

        # Rangebars
        top = max_
        bottom = min_
        left = x - 0.3
        right = x + 0.3
        # fig.segment(x0=x, y0=bottom, x1=x, y1=top,
        #             line_width=1.5, color='red')
        # fig.segment(x0=left, y0=top, x1=right, y1=top,
        #             line_width=1.5, color='red')
        # fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
        #             line_width=1.5, color='red')

        # Errorbars
        top = mean+stddev
        bottom = mean-stddev
        left = x-0.3
        right = x+0.3
        fig.segment(x0=x, y0=bottom, x1=x, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=top, x1=right, y1=top,
                    line_width=1.5, color='black')
        fig.segment(x0=left, y0=bottom, x1=right, y1=bottom,
                    line_width=1.5, color='black')

        self.layout = fig


class TFInputSelector(Component):
    name = 'TFInputSelector'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the TF of a selected tm, tmpix, cell

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

        self._tfinput = None
        self._vped = None

        self._tm = 0
        self._tmpix = 0
        self._cell = 0

        cdsource_d = dict(x=[], y=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)

        self.tm_select = None
        self.tmpix_select = None
        self.cell_select = None

    @property
    def tm(self):
        return self._tm

    @tm.setter
    def tm(self, val):
        self._tm = val
        self._update_cdsource()

    @property
    def tmpix(self):
        return self._tmpix

    @tmpix.setter
    def tmpix(self, val):
        self._tmpix = val
        self._update_cdsource()

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, val):
        self._cell = val
        self._update_cdsource()

    def _update_cdsource(self):
        x = self._vped
        y = self._tfinput[self.tm, self.tmpix, self.cell]
        a = np.where(y > 0)
        cdsource_d = dict(x=x, y=y)
        self.cdsource.data = cdsource_d
        # self.cdsource.trigger('data', None, None)
        # if self.layout:
        #     self.layout.children[1].children[0] = self.get_figure()

    def create(self, tf_input, vped):
        self._tfinput = tf_input
        self._vped = vped

        n_tm = tf_input.shape[0]
        n_tmpix = tf_input.shape[1]
        n_cells = tf_input.shape[2]

        self._update_cdsource()

        fig = self._get_figure()

        # Widgets
        tm_str = ['{:.0f}'.format(x) for x in range(n_tm)]
        tmpix_str = ['{:.0f}'.format(x) for x in range(n_tmpix)]
        cell_str = ['{:.0f}'.format(x) for x in range(n_cells)]

        self.tm_select = Select(title="TM:", value='0', options=tm_str)
        self.tmpix_select = Select(title="TMPIX:", value='0',
                                   options=tmpix_str)
        self.cell_select = Select(title="CELL:", value='0', options=cell_str)

        self.tm_select.on_change('value', self._on_tm_select)
        self.tmpix_select.on_change('value', self._on_tmpix_select)
        self.cell_select.on_change('value', self._on_cell_select)

        self.layout = layout([
            [self.tm_select, self.tmpix_select, self.cell_select],
            [fig]
        ])

    def _get_figure(self):
        title = "TF Selector"
        # tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        tools = "pan, box_zoom, wheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='wheel_zoom', webgl=True)
        c = fig.circle(source=self.cdsource, x='x', y='y', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                                renderers=[c]))
        # fig.y_range = Range1d(y_min, y_max, bounds=(y_min - 100, y_max + 100))
        # fig.x_range = Range1d(-100, n_cells + 100, bounds=(-500, n_cells + 500))
        fig.xaxis.axis_label = 'Vped'
        fig.yaxis.axis_label = 'ADC'

        return fig

    def _on_tm_select(self, attr, old, new):
        self.tm = int(self.tm_select.value)

    def _on_tmpix_select(self, attr, old, new):
        self.tmpix = int(self.tmpix_select.value)

    def _on_cell_select(self, attr, old, new):
        self.cell = int(self.cell_select.value)


class TargetCalibTFExplorer(Tool):
    name = "TargetCalibTFExplorer"
    description = "Plot the TargetCalib transfer function using bokeh"

    tfinput_path = Unicode(None, allow_none=True,
                           help='Path to a numpy file containing the input '
                                'TF array').tag(config=True)
    vped_path = Unicode(None, allow_none=True,
                        help='Path to a numpy file containing the vped '
                             'vector').tag(config=True)

    aliases = Dict(dict(tf='TFApplier.tf_path',
                        tf_input='TargetCalibTFExplorer.tfinput_path',
                        vped='TargetCalibTFExplorer.vped_path',
                        o='TargetCalibTFExplorer.output_dir'
                        ))
    classes = List([TFApplier
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tf = None

        self.p_tfspread = None
        self.p_tfselect = None
        self.p_tfinputspread = None
        self.p_tfinputselect = None

        self.layout = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        if (self.tfinput_path is None) != (self.vped_path is None):
            self.log.warn("Both tf_input and vped need to be supplied to "
                          "view the input TF array")

        self.tf = TFApplier(**kwargs)
        self.p_tfspread = TFSpread(**kwargs)
        self.p_tfselect = TFSelector(**kwargs)
        if self.tfinput_path and self.vped_path:
            self.p_tfinputspread = TFInputSpread(**kwargs)
            self.p_tfinputselect = TFInputSelector(**kwargs)

    def start(self):
        # Get TF
        tf, adc_min, adc_step = self.tf.get_tf()

        # Get TF Input
        tf_input = None
        vped = None
        if self.tfinput_path and self.vped_path:
            self.log.info("Loading TF input array file: {}"
                          .format(self.tfinput_path))
            tf_input = np.load(self.tfinput_path)
            self.log.info("Loading Vped vector file: {}"
                          .format(self.vped_path))
            vped = np.load(self.vped_path)

        # # Dimensions
        # n_tm = tf.shape[0]
        # n_tmpix = tf.shape[1]
        # n_cells = tf.shape[2]
        # n_points = tf.shape[3]

        # # Convert tm and tmpix to pixel
        # tf_pix = np.empty((n_tm * n_tmpix, n_cells, n_points))
        # tm_str = []
        # for tm in range(n_tm):
        #     tm_str.append('{:.0f}'.format(tm))
        #     for tmpix in range(n_tmpix):
        #         pix = pixels.convert_tm_tmpix_to_pix(tm, tmpix)
        #         tf_pix[pix] = tf[tm, tmpix]

        # Create bokeh figures
        self.p_tfspread.create(tf, adc_min, adc_step)
        self.p_tfselect.create(tf, adc_min, adc_step)
        if tf_input is not None and vped is not None:
            self.p_tfinputspread.create(tf_input, vped)
            self.p_tfinputselect.create(tf_input, vped)

        # Get bokeh layouts
        l_tfspread = self.p_tfspread.layout
        l_tfselect = self.p_tfselect.layout
        l_tfinputspread = None
        l_tfinputselect = None
        if self.p_tfinputspread:
            l_tfinputspread = self.p_tfinputspread.layout
            l_tfinputselect = self.p_tfinputselect.layout

        # Get widgets

        # Layout
        layout_list = [
            [l_tfspread],
            [l_tfselect]
        ]
        if l_tfinputspread:
            layout_list.append([l_tfinputspread])
        if l_tfinputselect:
            layout_list.append([l_tfinputselect])
        self.layout = layout(layout_list, sizing_mode="scale_width")

    def finish(self):
        fig_dir = join(dirname(self.tf.tf_path), "plot_tf")
        if not exists(fig_dir):
            self.log.info("Creating directory: {}".format(fig_dir))
            makedirs(fig_dir)

        path = join(fig_dir, 'tf.html')
        output_file(path)
        show(self.layout)
        self.log.info("Created bokeh figure: {}".format(path))

        curdoc().add_root(self.layout)
        curdoc().title = "Transfer Function"


exe = TargetCalibTFExplorer()
exe.run()
