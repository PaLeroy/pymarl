
REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .epsiode_runner_multi import EpisodeRunnerMulti
REGISTRY["episode_multi"] = EpisodeRunnerMulti

from .parallel_runner_multi import ParallelRunnerMulti
REGISTRY["parallel_multi"] = ParallelRunnerMulti
