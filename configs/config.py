from easydict import EasyDict

cfg = EasyDict()

cfg.steps_num = 1e4
cfg.arms_num = 10

cfg.run_experiments_series = False

# params for running series of experiments
cfg.envs = ['env-v0', 'env-v1', 'env-v2']
cfg.eps = [0.1, 0.01, 0.001]
cfg.init_types = ['optimistic', 'not_optimistic']
cfg.seeds = [0, 10, 1e2, 1e3, 1e4, 1e5]

# params for running single experiment
cfg.env_single = 'env-v0'
cfg.eps_single = 0.01
cfg.init_type_single = 'optimistic'

cfg.optimistic_init_value = 1e5
cfg.log_metrics = True

cfg.plots_dir = '../executor/plots/'
