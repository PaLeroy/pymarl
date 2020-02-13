
REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .epsiode_runner_multi import EpisodeRunnerMulti
REGISTRY["episode_multi"] = EpisodeRunnerMulti

from .parallel_runner_multi import ParallelRunnerMulti
REGISTRY["parallel_multi"] = ParallelRunnerMulti

from .epsiode_runner_population import EpisodeRunnerPopulation
REGISTRY["episode_runner_population"] = EpisodeRunnerPopulation

from .parallel_runner_population import ParallelRunnerPopulation
REGISTRY["parallel_runner_population"] = ParallelRunnerPopulation