from matchmaking.duo_matchmaking import DuoMatchmaking
from matchmaking.duo_random_matchmaking import DuoRandomMatchmaking
from matchmaking.random_diff_matchmaking import RandomDiffMatchmaking
from matchmaking.random_matchmaking import RandomMatchmaking
from matchmaking.single_matchmaking import SingleMatchmaking

REGISTRY = {"duo": DuoMatchmaking,
            "single": SingleMatchmaking,
            "duo_random": DuoRandomMatchmaking,
            "random": RandomMatchmaking,
            "random_diff": RandomDiffMatchmaking}

