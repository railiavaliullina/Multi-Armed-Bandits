
from configs.config import cfg
from e_greedy_algorithm.e_greedy import EGreedy


class Executor(object):

    @staticmethod
    def run_experiments_series():

        for env in cfg.envs:
            for epsilon in cfg.eps:
                for init_type in cfg.init_types:
                    for seed in cfg.seeds:
                        e_greedy = EGreedy(cfg, env_type=env, seed=seed, eps=epsilon, init_type=init_type)
                        e_greedy.run()
                        e_greedy.logger.end_logging()

    @staticmethod
    def run_single_experiment():

        for seed in cfg.seeds:
            e_greedy = EGreedy(cfg, env_type=cfg.env_single, seed=seed,
                               eps=cfg.eps_single, init_type=cfg.init_type_single)
            e_greedy.run()
            e_greedy.logger.end_logging()


if __name__ == '__main__':
    executor = Executor()
    if cfg.run_experiments_series:
        executor.run_experiments_series()
    else:
        executor.run_single_experiment()
