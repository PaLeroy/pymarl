from src.matchmaking.duo_matchmaking import DuoMatchmaking
from src.matchmaking.random_matchmaking import RandomMatchmaking
from src.matchmaking.single_matchmaking import SingleMatchmaking

REGISTRY = {"duo": DuoMatchmaking,
            "single": SingleMatchmaking,
            "random": RandomMatchmaking}

