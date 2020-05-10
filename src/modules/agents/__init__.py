

REGISTRY = {}

from .rnn_agent import RNNAgent
from .maven_rnn_agent import MavenRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["maven_rnn"] = MavenRNNAgent