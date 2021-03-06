import argparse
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ratel.agent import EpsilonGreedyAgent, GreedyAgent
from ratel.bandit import BernoulliBandit
from ratel.optimizer import Optimizer
from ratel.util import HarmonicStepper


def run_greedy(run_length, runs, start):
    rewards = np.arange(0.1, 0.2, 0.01)#.random.random(10)
    rewards[6] = 0.95
    rewards[3] = 0.94
    bandit = BernoulliBandit(rewards)
    agent = GreedyAgent(HarmonicStepper(length=10),
                        np.random.uniform(start - 1e-7, start + 1e-7, size=rewards.shape[0]))
    optimizer = Optimizer(agent, bandit)
    tot = pd.DataFrame({'wins': run_length * [0], 'rewards': run_length * [0]})
    for _ in range(runs):
        df = optimizer.run(run_length)
        tot += df
        optimizer.reset(start * np.ones(10))
    tot /= runs
    return tot


def run_e_greedy(run_length, runs, epsilon, start):
    rewards = np.arange(0.1, 0.2, 0.01)  # .random.random(10)
    rewards[6] = 0.95
    rewards[3] = 0.94
    bandit = BernoulliBandit(rewards)
    agent = EpsilonGreedyAgent(HarmonicStepper(length=10),
                               np.random.uniform(start - 1e-7, start + 1e-7, size=rewards.shape[0]),
                               epsilon=epsilon)
    optimizer = Optimizer(agent, bandit)
    tot = pd.DataFrame({'wins': run_length * [0], 'rewards': run_length * [0]})
    for _ in range(runs):
        df = optimizer.run(run_length)
        tot += df
        optimizer.reset(start * np.ones(10))
    tot /= runs
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--runs", required=True, type=int,
                    help="number of runs")
    ap.add_argument("-l", "--length", required=True, type=int,
                    help="length of run")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-g", "--greedy", action='store_true')
    group.add_argument("-e", "--epsilon", type=float, help="epsilon value for exploration")
    args = vars(ap.parse_args())
    starts = np.arange(0, 1.01, 0.1)
    pool = Pool()
    print(datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
    if args["greedy"]:
        greedy_dict = {s: pool.apply_async(run_greedy, (args["length"], args["runs"], s)) for s in starts}
        pool.close()
        pool.join()
        print(datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
        for s in greedy_dict:
            greedy_dict[s].get().to_csv(f"./greedy/greedy_extreme2_{s}.csv", index=False)
    elif "epsilon" in args:
        epsilon = args["epsilon"]
        epsilon_dict = {s: pool.apply_async(run_e_greedy, (args["length"], args["runs"], epsilon, s)) for s in starts}
        pool.close()
        pool.join()
        print(datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
        for s in epsilon_dict:
            epsilon_dict[s].get().to_csv(f"./epsilon/epsilon_extreme2_{epsilon}_{s}.csv", index=False)


if __name__ == '__main__':
    main()
