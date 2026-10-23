"""Explicit opt-in native equivariance components; existing MACE defaults are unchanged."""

from .channelwise import ChannelWiseTensorProduct, EdgeSchedule

__all__ = ["ChannelWiseTensorProduct", "EdgeSchedule"]
