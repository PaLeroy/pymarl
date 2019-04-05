REGISTRY = {}

from .basic_controller import BasicMAC
from .multi_team_controller import MultiMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["multi_mac"] = MultiMAC