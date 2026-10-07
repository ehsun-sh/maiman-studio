"""Built-in component library.

Components are ordinary Python classes. Third-party components install as normal
Python packages — no build step and no ABI to match, which is the difference
between a plugin system researchers will actually use and one they will not.
"""

from __future__ import annotations

from .amplifiers import EDFA
from .analyzers import (
    BERAnalyzer,
    ConstellationAnalyzer,
    ConstellationDiagram,
    EyeDiagram,
    Slicer,
)
from .coherent import (
    CarrierRecovery,
    CoherentReceiver,
    DualPolarizationReceiver,
    IQSampler,
)
from .couplers import EdgeCoupler, GratingCoupler
from .delay import DelayLine
from .detectors import APDPhotodiode, PINPhotodiode
from .dml import DirectlyModulatedLaser
from .dsp import (
    ButterflyEqualizer,
    CoarseFrequencyRecovery,
    DispersionCompensator,
    FrequencyRecovery,
    SoftDemapper,
    TimingRecovery,
)
from .electrical import (
    DCVoltage,
    FECDecoder,
    FECEncoder,
    IQDriver,
    NRZDriver,
    PRBSGenerator,
    SoftFECDecoder,
    SoftFECEncoder,
)
from .feedback import Feedback
from .fiber import Fiber
from .filters import ElectricalFilter, OpticalFilter, OpticalSpectrumAnalyzer
from .free_space import FreeSpaceChannel
from .line import AmplifiedLine
from .long_period import LongPeriodGrating
from .mapping import (
    DifferentialDecoder,
    PCSMapper,
    PilotInserter,
    PilotPhaseRecovery,
    QAMMapper,
)
from .meters import Oscilloscope, OSNRMeter, PowerMeter
from .modulators import IQModulator, MachZehnderModulator
from .otdr import BackscatterFiber
from .pam import FFEDFEEqualizer, PAM4Driver
from .passive import (
    Attenuator,
    Combiner,
    PolarizationCombiner,
    PolarizationRotator,
    Splitter,
)
from .photonic import (
    MMI,
    CoupledWaveguides,
    DirectionalCoupler,
    MachZehnderInterferometer,
    RingResonator,
    Waveguide,
)
from .quantum import BB84Receiver
from .reflective import Circulator, FiberBraggGrating
from .rf import ElectricalSpectrumAnalyzer, RFTone
from .sources import CWLaser, FabryPerotLaser, GaussianPulse, SechPulse, SweptLaser
from .tilted import TiltedFiberBraggGrating
from .wdm import Demultiplexer, Multiplexer, WavelengthSelectiveSwitch

__all__ = [
    "EDFA",
    "MMI",
    "APDPhotodiode",
    "AmplifiedLine",
    "Attenuator",
    "BB84Receiver",
    "BERAnalyzer",
    "BackscatterFiber",
    "ButterflyEqualizer",
    "CWLaser",
    "CarrierRecovery",
    "Circulator",
    "CoarseFrequencyRecovery",
    "CoherentReceiver",
    "Combiner",
    "ConstellationAnalyzer",
    "ConstellationDiagram",
    "CoupledWaveguides",
    "DCVoltage",
    "DelayLine",
    "Demultiplexer",
    "DifferentialDecoder",
    "DirectionalCoupler",
    "DirectlyModulatedLaser",
    "DispersionCompensator",
    "DualPolarizationReceiver",
    "EdgeCoupler",
    "ElectricalFilter",
    "ElectricalSpectrumAnalyzer",
    "EyeDiagram",
    "FECDecoder",
    "FECEncoder",
    "FFEDFEEqualizer",
    "FabryPerotLaser",
    "Feedback",
    "Fiber",
    "FiberBraggGrating",
    "FreeSpaceChannel",
    "FrequencyRecovery",
    "GaussianPulse",
    "GratingCoupler",
    "IQDriver",
    "IQModulator",
    "IQSampler",
    "LongPeriodGrating",
    "MachZehnderInterferometer",
    "MachZehnderModulator",
    "Multiplexer",
    "NRZDriver",
    "OSNRMeter",
    "OpticalFilter",
    "OpticalSpectrumAnalyzer",
    "Oscilloscope",
    "PAM4Driver",
    "PCSMapper",
    "PINPhotodiode",
    "PRBSGenerator",
    "PilotInserter",
    "PilotPhaseRecovery",
    "PolarizationCombiner",
    "PolarizationRotator",
    "PowerMeter",
    "QAMMapper",
    "RFTone",
    "RingResonator",
    "SechPulse",
    "Slicer",
    "SoftDemapper",
    "SoftFECDecoder",
    "SoftFECEncoder",
    "Splitter",
    "SweptLaser",
    "TiltedFiberBraggGrating",
    "TimingRecovery",
    "Waveguide",
    "WavelengthSelectiveSwitch",
]
