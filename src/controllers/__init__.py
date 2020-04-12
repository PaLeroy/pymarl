
REGISTRY = {}

from .basic_controller import BasicMAC
from .do_nothing_controller import DoNothingMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["do_not_mac"] = DoNothingMAC
