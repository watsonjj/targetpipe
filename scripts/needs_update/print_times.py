from traitlets import Dict, List
from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from datetime import datetime

def format_my_nanos(nanos):
    dt = datetime.fromtimestamp(nanos / 1e9)
    return '{}.{:09.0f}'.format(dt.strftime('%Y-%m-%dT%H:%M:%S'), nanos % 1e9)


class EventFileLooper(Tool):
    name = "EventFileLooper"
    description = "Loop through the file and apply calibration. Intended as " \
                  "a test that the routines work, and a benchmark of speed."

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        max_events='EventFileReaderFactory.max_events',
                        ))
    classes = List([EventFileReaderFactory,
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

    def start(self):
        source = self.file_reader.read()
        pt = 0
        ps = 0
        pn = 0
        pc = 0
        pd = 0
        pa = 0
        for event in source:
            t = event.meta['tack']
            s = event.meta['sec']
            n = event.meta['ns']
            c = s + n * 1e-9
            dt = datetime.fromtimestamp(c)
            at = event.trig.gps_time
            if event.count == 0:
                pt = t
                ps = s
                pn = n
                pc = c
                pd = dt
                pa = at

            print("INDEX: ", event.count)
            print("TACK: ", t, "({})".format(t-pt))
            print("SEC: ", s, "({})".format(s-ps))
            print("NS: ", n, "({})".format(n-pn))
            print("COM: ", c, "({})".format(c-pc))
            print("DT: ", dt, "({})".format(dt-pd))
            print("AT: ", at.iso, "({})".format((at-pa).sec))
            print()

            pt = t
            ps = s
            pn = n
            pc = c
            pd = dt
            pa = at

    def finish(self):
        pass


if __name__ == '__main__':
    exe = EventFileLooper()
    exe.run()
