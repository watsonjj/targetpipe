from ctapipe.core import Container, Item, Map
from ctapipe.io.containers import ReconstructedContainer, \
    CentralTriggerContainer, InstrumentContainer, \
    R0Container, R1Container, DL0Container, DL1Container
from numpy import ndarray
from astropy import units as u


class CHECR0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """
    adc_sums = Item(None, ("numpy array containing integrated ADC data "
                           "(n_channels x n_pixels)"))
    adc_samples = Item(None, ("numpy array containing ADC samples"
                              "(n_channels x n_pixels, n_samples)"))
    first_cell_ids = Item(ndarray, ("numpy array of the first_cell_id of each"
                                    "waveform in the camera image"))
    blockphase = Item(ndarray, ("numpy array of the blockphase of each "
                                "waveform in the camera image"))
    num_samples = Item(None, "number of time samples for telescope")


class CHECR0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """

    run_id = Item(-1, "run id number")
    event_id = Item(-1, "waveforms id number")
    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(CHECR0CameraContainer), "map of tel_id to "
                                           "CHECR0CameraContainer")


class CHECMCCameraEventContainer(Container):
    """
    Storage of mc data for a single telescope that change per event
    """
    reference_pulse_shape = Item(None, ("reference pulse shape for each "
                                        "channel"))
    time_slice = Item(0, "width of time slice", unit=u.ns)


class CHECMCEventContainer(Container):
    """
    Monte-Carlo
    """
    tel = Item(Map(CHECMCCameraEventContainer),
               "map of tel_id to MCCameraEventContainer")


class CHECDataContainer(Container):
    """ Top-level container for all waveforms information """

    r0 = Item(CHECR0Container(), "Raw Data")
    r1 = Item(R1Container(), "R1 Calibrated Data")
    dl0 = Item(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Item(DL1Container(), "DL1 Calibrated image")
    dl2 = Item(ReconstructedContainer(), "Reconstructed Shower Information")
    trig = Item(CentralTriggerContainer(), "central trigger information")
    count = Item(0, "number of events processed")
    inst = Item(InstrumentContainer(), "instrumental information (deprecated")
    mc = Item(CHECMCEventContainer(), "Monte-Carlo data")
