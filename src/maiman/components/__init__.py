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
from .detectors import APDPhotodiode, PINPhotodiode
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
from .mapping import (
    DifferentialDecoder,
    PilotInserter,
    PilotPhaseRecovery,
    QAMMapper,
)
from .meters import OSNRMeter, PowerMeter
from .modulators import IQModulator, MachZehnderModulator
from .passive import (
    Attenuator,
    Combiner,
    PolarizationCombiner,
    PolarizationRotator,
    Splitter,
)
from .photonic import (
    MMI,
    DirectionalCoupler,
    MachZehnderInterferometer,
    RingResonator,
    Waveguide,
)
from .reflective import Circulator, FiberBraggGrating
from .sources import CWLaser, GaussianPulse, SechPulse
from .wdm import Demultiplexer, Multiplexer

__all__ = [
    "EDFA",
    "MMI",
    "APDPhotodiode",
    "Attenuator",
    "BERAnalyzer",
    "ButterflyEqualizer",
    "CWLaser",
    "CarrierRecovery",
    "Circulator",
    "CoarseFrequencyRecovery",
    "CoherentReceiver",
    "Combiner",
    "ConstellationAnalyzer",
    "ConstellationDiagram",
    "DCVoltage",
    "Demultiplexer",
    "DifferentialDecoder",
    "DirectionalCoupler",
    "DispersionCompensator",
    "DualPolarizationReceiver",
    "ElectricalFilter",
    "EyeDiagram",
    "FECDecoder",
    "FECEncoder",
    "Feedback",
    "Fiber",
    "FiberBraggGrating",
    "FrequencyRecovery",
    "GaussianPulse",
    "IQDriver",
    "IQModulator",
    "IQSampler",
    "MachZehnderInterferometer",
    "MachZehnderModulator",
    "Multiplexer",
    "NRZDriver",
    "OSNRMeter",
    "OpticalFilter",
    "OpticalSpectrumAnalyzer",
    "PINPhotodiode",
    "PRBSGenerator",
    "PilotInserter",
    "PilotPhaseRecovery",
    "PolarizationCombiner",
    "PolarizationRotator",
    "PowerMeter",
    "QAMMapper",
    "RingResonator",
    "SechPulse",
    "Slicer",
    "SoftDemapper",
    "SoftFECDecoder",
    "SoftFECEncoder",
    "Splitter",
    "TimingRecovery",
    "Waveguide",
]
