import numpy as np
from tqdm import tqdm
from os import makedirs
from os.path import join, exists
from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import layout
from bokeh.models import HoverTool, ColumnDataSource
from traitlets import Dict, List
from ctapipe.core import Tool, Component
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory


class AdcSpread(Component):
    name = 'AdcSpread'

    def __init__(self, config, tool, **kwargs):
        """
        Bokeh plot for showing the spread of ADC for each event

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

    def create(self, mean, stddev, min_, max_):
        self.log.info("Creating {}".format(self.name))
        x = np.arange(mean.size)

        cdsource_d = dict(x=x, mean=mean, stddev=stddev, min=min_, max=max_)
        cdsource = ColumnDataSource(data=cdsource_d)

        title = "ADC Spread Vs Event"

        tools = "xpan, xwheel_pan, box_zoom, xwheel_zoom, save, reset"
        fig = figure(width=900, height=360, tools=tools, title=title,
                     active_scroll='xwheel_zoom', webgl=True)
        c = fig.circle(source=cdsource, x='x', y='mean', hover_color="red")
        fig.add_tools(HoverTool(tooltips=[("(x,y)", "(@x, @mean)"),
                                          ("stddev", "@stddev"),
                                          ("(min, max)", "@min, @max")
                                          ], renderers=[c]))

        fig.xaxis.axis_label = 'Event'
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


class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ped='CameraR1CalibratorFactory.pedestal_path',
                        tf='CameraR1CalibratorFactory.tf_path',
                        ))
    classes = List([EventFileReaderFactory,
                    CameraR1CalibratorFactory,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.r1 = None

        self.layout = None

        self.p_adcspread = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.p_adcspread = AdcSpread(**kwargs)

    def start(self):

        # Prepare storage array
        n_events = self.file_reader.num_events

        event_mean = np.zeros(n_events)
        event_stddev = np.zeros(n_events)
        event_min = np.zeros(n_events)
        event_max = np.zeros(n_events)

        source = self.file_reader.read()
        desc = "Looping through file"
        with tqdm(total=n_events, desc=desc) as pbar:
            for event in source:
                pbar.update(1)
                index = event.count

                self.r1.calibrate(event)

                telid = list(event.r0.tels_with_data)[0]
                r1 = event.r1.tel[telid].pe_samples[0]

                event_mean[index] = np.mean(r1)
                event_stddev[index] = np.std(r1)
                event_min[index] = np.min(r1)
                event_max[index] = np.max(r1)

        # Create bokeh figures
        self.p_adcspread.create(event_mean, event_stddev, event_min, event_max)

        # Get bokeh layouts
        l_adcspread = self.p_adcspread.layout

        # Layout
        layout_list = [
            [l_adcspread]
        ]
        self.layout = layout(layout_list, sizing_mode="scale_width")

    def finish(self):
        fig_dir = join(self.file_reader.output_directory, "plot_adc_drift")
        if not exists(fig_dir):
            self.log.info("Creating directory: {}".format(fig_dir))
            makedirs(fig_dir)

        path = join(fig_dir, 'adc_drift.html')
        output_file(path)
        show(self.layout)
        self.log.info("Created bokeh figure: {}".format(path))

        curdoc().add_root(self.layout)
        curdoc().title = "ADC Drift"


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
