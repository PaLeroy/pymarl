
REGISTRY = {}

from .basic_controller import BasicMAC
from .multi_team_controller import MultiMAC
from .do_nothing_controller import DoNothingMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["do_not_mac"] = DoNothingMAC
REGISTRY["multi_mac"] = MultiMAC