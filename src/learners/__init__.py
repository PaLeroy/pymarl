from .q_leaner_exec import QLearnerExec
from .q_learner import QLearner
from .coma_learner import COMALearner
from .coma_learner_multi import COMALearnerMulti
from .q_learner_multi import QLearnerMulti

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_exec"] = QLearnerExec
REGISTRY["coma_learner"] = COMALearner
REGISTRY["coma_learner_multi"] = COMALearnerMulti

REGISTRY["q_learner_multi"] = QLearnerMulti
