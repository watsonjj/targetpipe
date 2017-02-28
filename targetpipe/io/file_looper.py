from traitlets import List, Unicode, Int
from ctapipe.core import Component
from targetpipe.io.eventfilereader import TargetioFileReader


class TargetioFileLooper(Component):
    name = 'FileLooper'

    file_list = List(Unicode, None, allow_none=True,
                     help="List of event filepaths to read").tag(config=True)
    single_file = Unicode(None, allow_none=True,
                          help="Single file to read. If set then file_list is "
                               "ignored").tag(config=True)
    max_files = Int(None, allow_none=True,
                    help="Number of files to read").tag(config=True)
    max_events = Int(None, allow_none=True,
                     help="Max number of events to read from each "
                          "file").tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Loop through event files.

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

        if self.single_file is not None:
            self.file_list = [self.single_file]
        if not self.file_list:
            raise ValueError("Please specify at least one input filepath")

        self.file_reader_list = []
        for fn, fp in enumerate(self.file_list):
            if self.max_files is not None:
                if fn >= self.max_files:
                    break
            fr = TargetioFileReader(config, tool, input_path=fp,
                                    max_events=self.max_events)
            self.file_reader_list.append(fr)
        self.num_readers = len(self.file_reader_list)

    @property
    def num_events(self):
        n = 0
        for fn, file_reader in enumerate(self.file_reader_list):
            n += file_reader.num_events
        return n