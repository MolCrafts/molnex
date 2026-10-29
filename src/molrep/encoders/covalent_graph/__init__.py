"""Covalent-graph encoder: an edge-augmented graph transformer over the bonds, no coordinates."""

from molrep.encoders.covalent_graph.encoder import CovalentGraphEncoder
from molrep.encoders.covalent_graph.spec import CovalentGraphEncoderSpec

__all__ = ["CovalentGraphEncoder", "CovalentGraphEncoderSpec"]
