"""
Custom base boxplot module
"""
from astropy import log
import numpy as np
import numpy.ma as ma
from bokeh.models import HoverTool, ColumnDataSource


class Boxplot:
    """
    Custom boxplot that is flexible in which plotting library is requested.

    The description of a boxplot is taken from:
    http://www.physics.csbsju.edu/stats/box2.html

    Attributes
    ----------
    x_val : ndarray
        1D numpy array containging the x-axis values
    data2d : ndarray
        2D reshape of the input data array
    median : ndarray
        median for each x value
    upper_quartile : ndarray
        upper quartile for each x value
    lower_quartile : ndarray
        lower quartile for each x value
    upper_whisker : ndarray
        upper whisker for each x value
    lower_whisker : ndarray
        lower whisker for each x value
    suspected_outliers : ndarray
        masked numpy array containing the suspected outliers for each x value
    outliers : ndarray
        masked numpy array containing the outliers for each x value
    """
    def __init__(self, array, x_axis=None):
        """
        Parameters
        ----------
        array : ndarray
            2 dimensional numpy array, the first axis corresponding to the
            x-axis, and the second axis being the y-values that the boxplot
            statistics are calculated from.
        x_axis : ndarray
            1D axis of same size as the first dimension of array, containing
            the x axis values.
        """

        self.outliers_where = None
        self._calculate_stats(array, x_axis)

    def _calculate_stats(self, array, x_axis):
        s = array.shape
        ndim = array.ndim
        if ndim > 2:
            log.warning("Array dimensions are greater than 2")

        self.x_val = np.arange(s[0]) if x_axis is None else x_axis
        self.data2d = array

        # Statistics
        self.median = np.median(self.data2d, axis=1)
        self.upper_quartile = np.percentile(self.data2d, 75, axis=1)
        self.lower_quartile = np.percentile(self.data2d, 25, axis=1)
        iqr = self.upper_quartile - self.lower_quartile
        uif = self.upper_quartile[..., None] + 1.5 * iqr[..., None]
        lif = self.lower_quartile[..., None] - 1.5 * iqr[..., None]
        mge = ma.masked_greater_equal
        mle = ma.masked_less_equal
        self.upper_whisker = mge(self.data2d, uif).max(1).data
        self.lower_whisker = mle(self.data2d, lif).min(1).data
        uof = self.upper_quartile[..., None] + 3.0 * iqr[..., None]
        lof = self.lower_quartile[..., None] - 3.0 * iqr[..., None]
        self.suspected_outliers = ma.masked_where((self.data2d < uif) &
                                                  (self.data2d > lif) |
                                                  (self.data2d >= uof) |
                                                  (self.data2d <= lof),
                                                  self.data2d)
        self.outliers = ma.masked_where((self.data2d < uof) &
                                        (self.data2d > lof),
                                        self.data2d)
        self.suspected_outliers_where = np.where(~self.suspected_outliers.mask)
        self.outliers_where = np.where(~self.outliers.mask)
        self.upper_so_whisker = self.suspected_outliers.max(1).data
        self.lower_so_whisker = self.suspected_outliers.min(1).data
        self.upper_o_whisker = self.outliers.max(1).data
        self.lower_o_whisker = self.outliers.min(1).data

    def plot_bokeh(self, fig):
        x_diff = self.x_val[1] - self.x_val[0]
        widthq = x_diff * 0.4
        widthe = x_diff * 0.1
        radius = 7

        bottom = self.lower_quartile
        left = self.x_val - widthq
        top = self.upper_quartile
        right = self.x_val + widthq

        suspected_outliers_x = self.suspected_outliers_where[0]
        suspected_outliers_y = self.suspected_outliers[
            self.suspected_outliers_where].data
        outliers_x = self.outliers_where[0]
        outliers_y = self.outliers[self.outliers_where].data

        if suspected_outliers_y.size > (10 * self.x_val.size):
            count = self.suspected_outliers.count(axis=1)
            w = count > 10
            x = self.x_val[w]
            lower = self.lower_so_whisker[w]
            upper = self.upper_so_whisker[w]
            source = ColumnDataSource(dict(x=x, lower=lower, upper=upper,
                                           count=count[w]))
            fig.segment(x0=x, y0=lower, x1=x, y1=upper,
                        line_width=1.5, color='green')
            fig.segment(x0=x - widthe, y0=upper, x1=x + widthe, y1=upper,
                        line_width=1.5, color='green')
            fig.segment(x0=x - widthe, y0=lower, x1=x + widthe, y1=lower,
                        line_width=1.5, color='green')
            c1 = fig.circle('x', 'upper', size=radius, fill_alpha=0.4,
                            color='green', source=source)
            c2 = fig.circle('x', 'lower', size=radius, fill_alpha=0.4,
                            color='green', source=source)
            tooltips = [("(x, lower, upper)", "(@x, @lower, @upper)"),
                        ("count", "@count")]
            fig.add_tools(HoverTool(tooltips=tooltips, renderers=[c1, c2]))
        else:
            c1 = fig.circle(suspected_outliers_x, suspected_outliers_y,
                            size=radius, fill_alpha=0.4, color='green')
            fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                                    renderers=[c1]))
        if outliers_y.size > (10 * self.x_val.size):
            count = self.outliers.count(axis=1)
            w = count > 10
            x = self.x_val[w]
            lower = self.lower_o_whisker[w]
            upper = self.upper_o_whisker[w]
            source = ColumnDataSource(dict(x=x, lower=lower, upper=upper,
                                           count=count[w]))
            fig.segment(x0=x, y0=lower, x1=x, y1=upper,
                        line_width=1.5, color='red')
            fig.segment(x0=x - widthe, y0=upper, x1=x + widthe, y1=upper,
                        line_width=1.5, color='red')
            fig.segment(x0=x - widthe, y0=lower, x1=x + widthe, y1=lower,
                        line_width=1.5, color='red')
            c1 = fig.circle('x', 'upper', size=radius, fill_alpha=0.4,
                            color='red', source=source)
            c2 = fig.circle('x', 'lower', size=radius, fill_alpha=0.4,
                            color='red', source=source)
            tooltips = [("(x, lower, upper)", "(@x, @lower, @upper)"),
                        ("count", "@count")]
            fig.add_tools(HoverTool(tooltips=tooltips, renderers=[c1, c2]))
        else:
            c1 = fig.circle(outliers_x, outliers_y,
                            size=radius, fill_alpha=0.4, color='red')
            fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @y)")],
                                    renderers=[c1]))

        fig.quad(bottom=bottom, left=left, top=top, right=right,
                 fill_alpha=0.7, color="#B3DE69")
        fig.segment(x0=left, y0=self.median, x1=right, y1=self.median,
                    line_width=1.5)
        fig.segment(x0=self.x_val, y0=self.lower_whisker,
                    x1=self.x_val, y1=self.upper_whisker,
                    line_width=1.5, color='black')
        fig.segment(x0=self.x_val - widthe, y0=self.upper_whisker,
                    x1=self.x_val + widthe, y1=self.upper_whisker,
                    line_width=1.5, color='black')
        fig.segment(x0=self.x_val - widthe, y0=self.lower_whisker,
                    x1=self.x_val + widthe, y1=self.lower_whisker,
                    line_width=1.5, color='black')
