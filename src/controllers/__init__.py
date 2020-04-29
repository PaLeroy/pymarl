
REGISTRY = {}

from .basic_controller import BasicMAC
from .do_nothing_controller import DoNothingMAC
from .do_nothing_controller2 import DoNothingMAC2


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["do_not_mac"] = DoNothingMAC
REGISTRY["do_not_mac2"] = DoNothingMAC2
