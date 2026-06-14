"""Head module for molrep."""

from .labeler import Labeler, ProxyLabeler
from .scalar import ScalarHead
from .type import TypeHead

__all__ = ["TypeHead", "Labeler", "ProxyLabeler", "ScalarHead"]
