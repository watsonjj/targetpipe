import numpy as np
import pandas as pd
from pandas.computation.ops import UndefinedVariableError
from bokeh.layouts import row, widgetbox, column
from bokeh.models import Select, TextInput, Range1d
from bokeh.palettes import Spectral5
from bokeh.plotting import figure
from bokeh.charts import BoxPlot
from ctapipe.core import Component
from traitlets import CaselessStrEnum, Unicode


# TODO
"""
    Add:
        - 1D histogram
        - 2D histogram
        - Hovertool
        - Try chart scatter
"""


class ExploreDataApp(Component):
    """
    Class that utilises bokeh in order to explore and visualize a pandas
    DataFrame
    """
    name = 'ExploreDataApp'

    figures = ['scatter', 'boxplot']

    startup_figure = CaselessStrEnum(figures, 'scatter',
                                     help='Figure to plot '
                                          'first').tag(config=True)
    startup_query = Unicode('', allow_none=True,
                            help='Pandas query to apply to the DataFrame '
                                 'before plotting').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Parent class for the dl0 data volume reducers. Fills the dl0 container.

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

        self.base_df = None
        self.df = None
        self.sizes = list(range(6, 22, 3))
        self.colors = Spectral5
        self.columns = None
        self.discrete = None
        self.continuous = None
        self.quantileable = None
        self.query_history_list = None
        self.wb = None
        self.fig = None
        self.layout = None

    def start(self, df):
        self.base_df = df
        self.df = self.base_df
        if self.startup_query:
            self.df = self.base_df.query(self.startup_query)

        self.columns = sorted(df.columns)
        self.discrete = [x for x in self.columns if df[x].dtype == object]
        self.continuous = [x for x in self.columns if x not in self.discrete]
        self.quantileable = [x for x in self.continuous if
                             len(df[x].unique()) > 20]

        self.query_history_list = [self.startup_query]
        init_wb = self._create_init_widgets(self.startup_figure,
                                            self.startup_query)

        self.wb = self._create_widgets()
        self.fig = self._create_figure()
        self.layout = row(column(init_wb, self.wb), self.fig)

    def _create_init_widgets(self, startup_figure, startup_query):
        self.w_figure = Select(title='Figure Type', value=startup_figure,
                               options=self.figures)
        self.w_figure.on_change('value', self.update_canvas)

        self.w_query = TextInput(title="Query", value=startup_query)
        self.w_query.on_change('value', self.update_on_query_change)

        self.w_query_history = Select(title='Query History',
                                      value=startup_query,
                                      options=self.query_history_list)
        self.w_query_history.on_change('value',
                                       self.update_on_query_history_change)

        init_wb = widgetbox([self.w_figure, self.w_query,
                             self.w_query_history], width=200)

        return init_wb

    def _create_figure(self):
        fig = None
        if self.w_figure.value == 'scatter':
            fig = self._create_scatter()
        elif self.w_figure.value == 'boxplot':
            fig = self._create_boxplot()

        return fig

    def _create_widgets(self):
        wb = None
        if self.w_figure.value == 'scatter':
            wb = self._create_scatter_wb()
        elif self.w_figure.value == 'boxplot':
            wb = self._create_boxplot_wb()
        return wb

    def _create_scatter(self):
        xs = self.df[self.w_x.value].values
        ys = self.df[self.w_y.value].values
        x_title = self.w_x.value.title()
        y_title = self.w_y.value.title()

        kw = dict()
        if self.w_x.value in self.discrete:
            kw['x_range'] = sorted(set(xs))
        if self.w_y.value in self.discrete:
            kw['y_range'] = sorted(set(ys))
        kw['title'] = "%s vs %s" % (x_title, y_title)

        fig = figure(plot_height=600, plot_width=800,
                     tools='pan,box_zoom,reset',
                     webgl=(self.w_webgl.value == 'True'), **kw)
        fig.xaxis.axis_label = x_title
        fig.yaxis.axis_label = y_title

        if self.w_x.value in self.discrete:
            fig.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if self.w_size.value != 'None':
            groups = pd.qcut(self.df[self.w_size.value].values,
                             len(self.sizes))
            sz = [self.sizes[xx] for xx in groups.codes]

        c = "#31AADE"
        if self.w_color.value != 'None':
            groups = pd.qcut(self.df[self.w_color.value].values,
                             len(self.colors))
            c = [self.colors[xx] for xx in groups.codes]
        fig.circle(x=xs, y=ys, color=c, size=sz, line_color="white",
                   alpha=0.6,
                   hover_color='white', hover_alpha=0.5)

        return fig

    def _create_scatter_wb(self):
        self.w_x = Select(title='X-Axis', value='file_num', options=self.columns)
        self.w_x.on_change('value', self.update_figure)

        self.w_y = Select(title='Y-Axis', value='charge', options=self.columns)
        self.w_y.on_change('value', self.update_figure)

        self.w_color = Select(title='Color', value='None',
                              options=['None'] + self.quantileable)
        self.w_color.on_change('value', self.update_figure)

        self.w_size = Select(title='Size', value='None',
                             options=['None'] + self.quantileable)
        self.w_size.on_change('value', self.update_figure)

        self.w_webgl = Select(title='Webgl', value='True',
                              options=['True', 'False'])
        self.w_webgl.on_change('value', self.update_figure)

        wb = widgetbox([self.w_x, self.w_y, self.w_color, self.w_size,
                        self.w_webgl], width=200)
        return wb

    def _create_boxplot(self):
        print("boxplot")
        ys = self.df[self.w_y.value].values
        x_title = self.w_x.value.title()
        y_title = self.w_y.value.title()

        kw = dict()
        if self.w_y.value not in self.discrete:
            min_y = np.min(ys)
            max_y = np.max(ys)
            pad_y = (max_y - min_y) * 0.05
            kw['y_range'] = Range1d(min_y - pad_y, max_y + pad_y)
        kw['title'] = "%s vs %s (boxplot)" % (x_title, y_title)

        p = BoxPlot(self.df, values=self.w_y.value, label=self.w_x.value,
                    color=self.w_color.value,
                    whisker_color=self.w_whisker.value,
                    plot_height=600, plot_width=800, legend=False,
                    tools='pan,box_zoom,reset', **kw)

        if 'y_range' in kw:
            p.y_range = kw['y_range']

        return p

    def _create_boxplot_wb(self):
        print("boxplot_wb")
        self.w_x = Select(title='X-Axis', value='file', options=self.columns)
        self.w_x.on_change('value', self.update_figure)

        self.w_y = Select(title='Y-Axis', value='charge',
                          options=self.quantileable)
        self.w_y.on_change('value', self.update_figure)

        self.w_color = Select(title='Color', value='blue',
                              options=['blue'] + self.quantileable)
        self.w_color.on_change('value', self.update_figure)

        self.w_whisker = Select(title='Whisker Color', value='black',
                                options=['black'] + self.quantileable)
        self.w_whisker.on_change('value', self.update_figure)

        # wb = widgetbox([self.w_x, self.w_y, self.w_color, self.w_whisker,
        #                 self.w_query], width=200)
        wb = widgetbox([self.w_x, self.w_y], width=200)
        return wb

    def update_figure(self, attr, old, new):
        self.layout.children[1] = self._create_figure()

    def update_canvas(self, attr, old, new):
        self.layout.children[0].children[1] = self._create_widgets()
        self.update_figure(attr, old, new)

    def update_on_query_history_change(self, attr, old, new):
        self.w_query.value = self.w_query_history.value

    def update_on_query_change(self, attr, old, new):
        try:
            self.df = self.base_df.query(self.w_query.value)
        except UndefinedVariableError:
            print("UndefinedVariableError")
            return
        except ValueError:
            print("ValueError")
            self.df = self.base_df
            self.update_canvas(attr, old, new)
            return

        if self.w_query.value not in self.query_history_list:
            self.query_history_list.append(self.w_query.value)
            self.w_query_history.options = self.query_history_list
        self.update_figure(attr, old, new)
