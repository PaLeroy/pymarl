from matchmaking.duo_matchmaking import DuoMatchmaking
from matchmaking.duo_random_matchmaking import DuoRandomMatchmaking
from matchmaking.random_diff_matchmaking import RandomDiffMatchmaking
from matchmaking.random_matchmaking import RandomMatchmaking
from matchmaking.single_matchmaking import SingleMatchmaking
from matchmaking.duo_fair import DuoFair

REGISTRY = {"duo": DuoMatchmaking,
            "duo_fair": DuoFair,
            "single": SingleMatchmaking,
            "duo_random": DuoRandomMatchmaking,
            "random": RandomMatchmaking,
            "random_diff": RandomDiffMatchmaking}

