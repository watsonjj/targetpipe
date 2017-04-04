import numpy as np
from bokeh.models import ColumnDataSource, Button
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import layout
from ctapipe.core import Tool
from targetpipe.plots.event_viewer import EventViewer
from targetpipe.visualization.bokeh import FastCameraDisplay
from targetpipe.io.pixels import pixel_sizes

import waveform_thread

PLOTARGS = dict(tools="", toolbar_location=None,
                outline_line_color='#595959', webgl=True)

# filename = join(dirname(__file__), "description.html")
# desc = Div(text=open(filename).read(), render_as_text=False, width=1000)


class HillasViewer:
    def __init__(self, hillas_edges_dict):
        self.s_dict = {}
        self.f_dict = {}
        self.h_dict = {}
        for param, edges in hillas_edges_dict.items():
            zeros = np.zeros(edges.size-1)
            left = edges[:-1]
            right = edges[1:]
            source = ColumnDataSource(data=dict(top=zeros,
                                                left=left,
                                                right=right,
                                                bottom=zeros))

            fig = figure(plot_width=200, plot_height=200,
                         # x_range=(0, 100), y_range=(0, 100),
                         title=param, **PLOTARGS)
            fig.quad(bottom='bottom', left='left', right='right',
                     top='top', source=source, alpha=0.5)

            self.s_dict[param] = source
            self.f_dict[param] = fig

        d = self.f_dict
        self.layout = layout([
            [d['width'], d['length'], d['size']],
            [d['phi'], d['miss'], d['r']]
        ])

    def update(self):
        hillas_dict = waveform_thread.HILLAS
        for param, vals in hillas_dict.items():
            self.s_dict[param].data['top'] = vals
            # Need to trigger to refresh plot (when update with numpy)????
            self.s_dict[param].trigger('data', None, None)


class BokehLiveCamera(Tool):
    name = "BokehLiveCamera"
    description = "Display camera images and hillas histograms live"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._freeze_data = None

        self.live_viewer = None
        self.freeze_viewer = None
        self.hillas_viewer = None

        self.layout = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        # Wait for waveform thread
        while waveform_thread.PIXEL_POS is None:
            pass
        x_pix_pos, y_pix_pos = waveform_thread.PIXEL_POS.value
        pix_sizes = pixel_sizes(x_pix_pos, y_pix_pos)

        self.live_viewer = FastCameraDisplay(x_pix_pos, y_pix_pos, pix_sizes)
        # figure(plot_width=400, plot_height=400, **PLOTARGS)
        self.freeze_viewer = EventViewer(**kwargs)
        self.freeze_viewer.create()
        self.freeze_viewer.enable_automatic_index_increment()
        self.hillas_viewer = HillasViewer(waveform_thread.HILLAS_EDGES)

        # Widgets
        w_freeze = Button(label="Get Event", button_type="success")
        w_freeze.on_click(self.get_freeze_event)

        # Setup layout
        self.layout = layout([
            [self.live_viewer.layout, self.hillas_viewer.layout],
            [self.freeze_viewer.layout],
            [w_freeze]
        ])

    def start(self):
        pass

    def finish(self):
        curdoc().add_periodic_callback(self.live_update, 40)
        curdoc().add_periodic_callback(self.hillas_update, 200)
        curdoc().add_root(self.layout)
        curdoc().title = "Live Camera"

    def live_update(self):
        self.live_viewer.image = waveform_thread.LIVE_DATA['image']

    def get_freeze_event(self):
        print("getting freeze")
        self.freeze_data = waveform_thread.FREEZE_DATA

    @property
    def freeze_data(self):
        return self._freeze_data

    @freeze_data.setter
    def freeze_data(self, val):
        self._freeze_data = val
        self.freeze_viewer.event = val['event']

    def hillas_update(self):
        self.hillas_viewer.update()

exe = BokehLiveCamera()
exe.run()
