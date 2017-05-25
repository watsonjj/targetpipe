"""
Handles reading of different event/waveform containing files
"""

import numpy as np
from copy import deepcopy
from traitlets import Unicode, List, Int, observe
from ctapipe.core import Component
from ctapipe.io.eventfilereader import EventFileReader
from targetpipe.io.targetio import TargetioExtractor
from targetpipe.io.toyio import toyio_event_source, toyio_get_num_events


class TargetioFileReader(EventFileReader):
    name = 'TargetioFileReader'
    origin = 'targetio'

    input_path = Unicode(None, allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Class to handle targetio input files. Enables obtaining the "source"
        generator.

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
        super().__init__(config=config, tool=tool, **kwargs)
        self.extractor = TargetioExtractor(self.input_path, self.max_events)

    @observe('input_path')
    def on_input_path_changed(self, change):
        new = change['new']
        try:
            self.log.warning("Change: input_path={}".format(new))
            self._num_events = None
            self._event_id_list = []
            self._init_path(new)
            self.extractor = TargetioExtractor(new, self.max_events)
        except AttributeError:
            pass

    @staticmethod
    def check_file_compatibility(file_path):
        compatible = True
        # TODO: Change check to be a try of targetio_event_source?
        if not file_path.endswith('.fits') and not file_path.endswith('.tio'):
            compatible = False
        return compatible

    @property
    def num_events(self):
        if not self._num_events:
            # self.log.info("Obtaining number of events in file...")
            num = self.extractor.n_events
            if self.max_events and self.max_events < num:
                num = self.max_events
            # self.log.info("Number of events = {}".format(num))
            self._num_events = num
        return self._num_events

    @property
    def event_id_list(self):
        if not self._event_id_list:
            # self.log.info("Building new list of event ids...")
            self._event_id_list = self.extractor.event_id_list[:self.max_events]
            # self.log.info("List of event ids built.")
        return self._event_id_list

    def read(self, allowed_tels=None, requested_event=None,
             use_event_id=False):
        """
        Read the file using the appropriate method depending on the file origin

        Parameters
        ----------
        allowed_tels : list[int]
            select only a subset of telescope, if None, all are read. This can
            be used for example emulate the final CTA data format, where there
            would be 1 telescope per file (whereas in current monte-carlo,
            they are all interleaved into one file)
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index

        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

        # Obtain relevent source
        self.log.debug("Reading file...")
        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))
        source = self.extractor.read_generator()
        self.log.debug("File reading complete")
        return source

    def get_event(self, requested_event, use_event_id=False):
        """
        Loop through events until the requested event is found

        Parameters
        ----------
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index

        Returns
        -------
        event : `ctapipe` event-container

        """
        self.extractor.read_event(requested_event, use_event_id)
        event = self.extractor.data
        return deepcopy(event)


class ToyioFileReader(EventFileReader):
    name = 'ToyioFileReader'
    origin = 'toyio'

    input_path = Unicode(None, allow_none=True,
                         help='Path to the input file containing '
                              'events.').tag(config=True)

    @staticmethod
    def check_file_compatibility(file_path):
        compatible = True
        # TODO: Change check to be a try of targetio_event_source?
        if not file_path.endswith('.npy'):
            compatible = False
        return compatible

    @property
    def num_events(self):
        # self.log.info("Obtaining number of events in file...")
        if not self._num_events:
            num = toyio_get_num_events(self.input_path,
                                       max_events=self.max_events)
            self._num_events = num
        # self.log.info("Number of events inside file = {}"
        #               .format(self._num_events))
        return self._num_events

    @property
    def event_id_list(self):
        self.log.info("Retrieving list of event ids...")
        if self._event_id_list:
            pass
        else:
            self.log.info("Building new list of event ids...")
            self._event_id_list = np.arange(self.num_events)
        self.log.info("List of event ids retrieved.")
        return self._event_id_list

    def read(self, allowed_tels=None, requested_event=None,
             use_event_id=False):
        """
        Read the file using the appropriate method depending on the file origin

        Parameters
        ----------
        allowed_tels : list[int]
            select only a subset of telescope, if None, all are read. This can
            be used for example emulate the final CTA data format, where there
            would be 1 telescope per file (whereas in current monte-carlo,
            they are all interleaved into one file)
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index

        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

        # Obtain relevent source
        self.log.debug("Reading file...")
        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))
        source = toyio_event_source(self.input_path,
                                    max_events=self.max_events,
                                    allowed_tels=allowed_tels,
                                    requested_event=requested_event,
                                    use_event_id=use_event_id)
        self.log.debug("File reading complete")
        return source
