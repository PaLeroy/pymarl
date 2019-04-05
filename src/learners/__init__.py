from .q_learner import QLearner
from .coma_learner import COMALearner
from .q_learner_multi import QLearnerMulti

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["q_learner_multi"] = QLearnerMulti
