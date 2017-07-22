"""Module to handle to storage of events extracted with `target_io` into
containers defined in `ctapipe.io.containers`.
"""

import numpy as np

from astropy import units as u
from astropy.time import Time

from targetpipe.io.containers import CHECDataContainer as DataContainer
from targetpipe.io.camera import Config


def toyio_get_num_events(url, max_events=None):
    """
    Faster method to get the number of events that exist in the file.

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read

    Returns
    -------
    n_events : int

    """
    waveforms_cells = np.load(url)
    n_events = waveforms_cells.shape[0]
    if max_events is not None and n_events > max_events:
        n_events = max_events
    return n_events


def toyio_event_source(url, max_events=None, allowed_tels=None,
                       requested_event=None, use_event_id=False):
    """A generator that streams data from an EventIO/HESSIO MC data file
    (e.g. a standard CTA data file.)

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : list[int]
        select only a subset of telescope, if None, all are read. This can
        be used for example emulate the final CTA data format, where there
        would be 1 telescope per file (whereas in current monte-carlo,
        they are all interleaved into one file)
    requested_event : int
        Seek to a paricular waveforms index
    use_event_id : bool
        If True ,'requested_event' now seeks for a particular waveforms id
        instead of index
    """

    waveforms_cells = np.load(url)
    waveforms = waveforms_cells[:, :, 1, :]
    cells = waveforms_cells[:, :, 0, :]
    n_events, n_pix, _, n_samples = waveforms_cells.shape
    chec_tel = 0
    run_id = 0
    cameraconfig = Config()
    pix_pos = cameraconfig.pixel_pos

    counter = 0
    if allowed_tels is not None:
        allowed_tels = set(allowed_tels)
    data = DataContainer()
    data.meta['source'] = "toyio"

    # some hessio_event_source specific parameters
    data.meta['input'] = url
    data.meta['max_events'] = max_events
    data.meta['num_events'] = n_events

    for event_id in range(n_events):

        # Seek to requested waveforms
        if requested_event is not None:
            current = counter
            if use_event_id:
                current = event_id
            if not current == requested_event:
                counter += 1
                continue

        data.r0.run_id = run_id
        data.r0.event_id = event_id
        data.r0.tels_with_data = {chec_tel}
        data.r1.run_id = run_id
        data.r1.event_id = event_id
        data.r1.tels_with_data = {chec_tel}
        data.dl0.run_id = run_id
        data.dl0.event_id = event_id
        data.dl0.tels_with_data = {chec_tel}

        # handle telescope filtering by taking the intersection of
        # tels_with_data and allowed_tels
        if allowed_tels is not None:
            selected = data.dl0.tels_with_data & allowed_tels
            if len(selected) == 0:
                continue  # skip waveforms
            data.dl0.tels_with_data = selected

        data.trig.tels_with_trigger = [chec_tel]
        time_ns = counter
        data.trig.gps_time = Time(time_ns * u.ns,
                                  format='gps', scale='utc')

        data.count = counter

        # this should be done in a nicer way to not re-allocate the
        # data each time (right now it's just deleted and garbage
        # collected)

        data.r0.tel.clear()
        data.r1.tel.clear()
        data.dl0.tel.clear()
        data.dl1.tel.clear()

        data.inst.pixel_pos[chec_tel] = pix_pos
        data.inst.optical_foclen[chec_tel] = 2.283 * u.m
        data.inst.num_channels[chec_tel] = 1
        data.inst.num_pixels[chec_tel] = n_pix
        # data.inst.num_samples[chec_tel] = n_samples

        # waveforms.mc.tel[tel_id] = MCCameraContainer()

        # load the data per telescope/chan
        data.r0.tel[chec_tel].adc_samples = \
            waveforms[event_id, :, :][None, ...]
        # data.r0.tel[chec_tel].adc_sums = \
        #     targetio_extractor.waveforms.sum(axis=1)[None, ...]
        data.r1.tel[chec_tel].pe_samples = \
            np.zeros(waveforms[0].shape)[None, ...]
        data.r0.tel[chec_tel].first_cell_ids = \
            cells[event_id, :, 0]

        yield data
        counter += 1

        if max_events is not None and counter >= max_events:
            break
