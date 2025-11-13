from nni.experiment import Experiment
import json
import argparse


parser = argparse.ArgumentParser("AutoML Hyperparameter Tuning for FL Algorithm")
parser.add_argument("--dataset", default="Office-Caltech", type=str, help="Mnist, Cifar10, Digits, Office-Caltech")
parser.add_argument("--algorithm", default="Fedavg", type=str)
parser.add_argument("--dynamic_type", default="static", type=str)
parser.add_argument("--num_clients", default=100, type=int, help="number of clients")
parser.add_argument("--alpha", default=0.1, type=float, help="concentration level of Dirichlet distribution")
parser.add_argument("--kpfl", default=1, type=int, help="dynamic participation for federated learning")
parser.add_argument("--port", default=8888, type=int)
args = parser.parse_args()


''' Define search space '''
with open('./configs/search_space.json', 'r') as f:
    search_space = json.load(f)

algo = args.algorithm
if args.kpfl:
    search_space = search_space["fedkpfl"]
else:
    search_space = search_space[algo]
print(search_space)


''' Run experiment '''  
experiment = Experiment('local')
experiment.config.experiment_name = "({}, {} (KPFL)) {}, nc={} ({})-Tuning".format(args.dataset, args.alpha, algo, args.num_clients, args.dynamic_type)
experiment.config.trial_command = 'python3 main.py --dataset {} --algorithm {} --num_clients {} --skew_type label --alpha {} --dynamic_type {} --dpfl {} --nni'.format(args.dataset, algo.lower(), args.num_clients, args.alpha, args.dynamic_type, args.dpfl)
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 2
experiment.run(args.port)

input('Press enter to quit...')
experiment.stop()
# experiment.view()