from src.learners.v_learner import VLearner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .coma_learner_multi import COMALearnerMulti
from .q_learner_multi import QLearnerMulti

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["v_learner"] = VLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["coma_learner_multi"] = COMALearnerMulti

REGISTRY["q_learner_multi"] = QLearnerMulti
