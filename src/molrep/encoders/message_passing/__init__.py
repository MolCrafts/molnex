"""Message-passing encoder: atom and bond streams over the covalent bonds, no coordinates."""

from molrep.encoders.message_passing.encoder import MessagePassingEncoder
from molrep.encoders.message_passing.spec import DiscreteChannel, MessagePassingEncoderSpec

__all__ = ["DiscreteChannel", "MessagePassingEncoder", "MessagePassingEncoderSpec"]
