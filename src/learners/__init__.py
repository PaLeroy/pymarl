from .do_not_learn import DoNotLearn
from .q_leaner_exec import QLearnerExec
from .q_learner import QLearner
from .coma_learner import COMALearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_exec"] = QLearnerExec
REGISTRY["coma_learner"] = COMALearner
REGISTRY["do_not_learn"] = DoNotLearn
