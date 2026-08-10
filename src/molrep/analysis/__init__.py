"""molrep.analysis — latent atom embedding diagnostics (Validation D/E light).

Must not import molpot / molzoo / condensation.
"""

from molrep.analysis.artifacts import LatentAnalysisArtifacts
from molrep.analysis.latent_store import AtomLatentTable
from molrep.analysis.projection import LatentPCA2D
from molrep.analysis.type_purity import NearestNeighbourTypePurity, TypePurityReport

__all__ = [
    "AtomLatentTable",
    "NearestNeighbourTypePurity",
    "TypePurityReport",
    "LatentPCA2D",
    "LatentAnalysisArtifacts",
]
