"""
Custom base boxplot module
"""
import numpy as np
import numpy.ma as ma
from bokeh.models import HoverTool, ColumnDataSource


class Boxplot:
    """
    Custom boxplot that is flexible in which plotting library is requested.

    The description of a boxplot is taken from:
    http://www.physics.csbsju.edu/stats/box2.html
    """

    def __init__(self, fig):
        cdsource_d = dict(x=[],
                          median=[],
                          mean=[],
                          top=[],
                          bottom=[],
                          left=[],
                          right=[],
                          emax=[],
                          emin=[],
                          el=[],
                          er=[])
        self.cdsource = ColumnDataSource(data=cdsource_d)
        cdsource_outliers_d = dict(x=[],
                                   y=[],
                                   i=[],
                                   color=[])
        self.cdsource_outliers = ColumnDataSource(data=cdsource_outliers_d)


        fig.quad(source=self.cdsource,
                 bottom='bottom', left='left', top='top', right='right',
                 fill_alpha=0.4, color="#000099")
        fig.segment(source=self.cdsource,
                    x0='left', y0='median', x1='right', y1='median',
                    line_width=1.5, color='red')
        fig.circle(source=self.cdsource, x='x', y='mean', color='purple')
        fig.segment(source=self.cdsource,
                    x0='x', y0='top', x1='x', y1='emax',
                    line_width=1.5, color='black')
        fig.segment(source=self.cdsource,
                    x0='el', y0='emax', x1='er', y1='emax',
                    line_width=1.5, color='black')
        fig.segment(source=self.cdsource,
                    x0='x', y0='emin', x1='x', y1='bottom',
                    line_width=1.5, color='black')
        fig.segment(source=self.cdsource,
                    x0='el', y0='emin', x1='er', y1='emin',
                    line_width=1.5, color='black')
        c = fig.circle(source=self.cdsource_outliers, x='x', y='y',
                       fill_alpha=0.6, color='color', radius=7)

        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)"),
                                          ("i", "@i")],
                                renderers=[c]))

    def update(self, x, y_data):
        # Statistics
        median = np.median(y_data, axis=1)
        mean = np.mean(y_data, axis=1)
        upper_quartile = np.percentile(y_data, 75, axis=1)
        lower_quartile = np.percentile(y_data, 25, axis=1)
        iqr = upper_quartile - lower_quartile
        uif = upper_quartile[..., None] + 1.5 * iqr[..., None]
        lif = lower_quartile[..., None] - 1.5 * iqr[..., None]
        mge = ma.masked_greater_equal
        mle = ma.masked_less_equal
        upper_whisker = mge(y_data, uif).max(1).data
        lower_whisker = mle(y_data, lif).min(1).data
        uof = upper_quartile[..., None] + 3.0 * iqr[..., None]
        lof = lower_quartile[..., None] - 3.0 * iqr[..., None]
        susp_outliers = ma.masked_where((y_data < uif) & (y_data > lif) |
                                        (y_data >= uof) | (y_data <= lof),
                                        y_data)
        outliers = ma.masked_where((y_data < uof) & (y_data > lof), y_data)
        susp_outliers_where = np.where(~susp_outliers.mask)
        outliers_where = np.where(~outliers.mask)

        x_diff = x[1] - x[0]
        widthq = x_diff * 0.4
        widthe = x_diff * 0.1

        bottom = lower_quartile
        left = x - widthq
        top = upper_quartile
        right = x + widthq

        el = x - widthe
        er = x + widthe

        suspected_outliers_x = susp_outliers_where[0]
        suspected_outliers_i = susp_outliers_where[1]
        suspected_outliers_y = susp_outliers[susp_outliers_where].data
        outliers_x = outliers_where[0]
        outliers_i = outliers_where[1]
        outliers_y = outliers[outliers_where].data

        all_outliers_x = x[np.append(suspected_outliers_x, outliers_x)]
        all_outliers_y = np.append(suspected_outliers_y, outliers_y)
        all_outliers_i = np.append(suspected_outliers_i, outliers_i)
        color = np.full(all_outliers_x.shape, "red", dtype='<U5')
        color[:suspected_outliers_x.size] = "green"

        cdsource_d = dict(x=x,
                          median=median,
                          mean=mean,
                          top=top,
                          bottom=bottom,
                          left=left,
                          right=right,
                          emax=upper_whisker,
                          emin=lower_whisker,
                          el=el,
                          er=er)
        self.cdsource.data = cdsource_d
        cdsource_outliers_d = dict(x=all_outliers_x,
                                   y=all_outliers_y,
                                   i=all_outliers_i,
                                   color=color)
        self.cdsource_outliers.data = cdsource_outliers_d
