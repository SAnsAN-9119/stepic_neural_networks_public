# run_car.py
import sys
sys.path.append('D:/Users/sansa/Programming/Study/Neural_networks')
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
import numpy as np
import random
import os
import json

import argparse

def load_world(seed, visualize, agent_filename=None):
    np.random.seed(seed)
    random.seed(seed)
    m = generate_map(8, 5, 3, 3)
    if agent_filename:
        agent = SimpleCarAgent.from_file(agent_filename)
    else:
        agent = SimpleCarAgent()
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2, visualize=visualize)
    w.set_agents([agent])
    return w

def test_agent(seed, filename, steps=800):
    testw = load_world(seed, False, filename)
    agent = testw.agents[0]
    try:
        return testw.evaluate_agent(agent, steps, visual=False)
    except:
        return -1000

def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, help="Количество шагов",  default=1600)
    parser.add_argument("--seed", type=int, default=7, help="Случайное зерно")
    parser.add_argument("-e", "--evaluate", action='store_true', help="Режим оценки",  default=False)
    parser.add_argument("-v", "--visualize", action='store_true', help="Включить визуализацию через Pygame", default=False)
    parser.add_argument("-f", "--filename", type=str, default=None, help="Имя файла с весами")
    # parser.add_argument("-f", "--filename", type=str, default='network_config_agent_0_layers_13_32_64_32_1.txt')
    # parser.add_argument("-s", "--steps", type=int, default=800)
    # parser.add_argument("--seed", type=int, default=3)
    # parser.add_argument("-e", "--evaluate", type=bool, default=True)
    # parser.add_argument("-v", "--visualize", action='store_true', help="Включить визуализацию через Pygame", default=True)
    args = parser.parse_args()

    print(args.steps, args.seed, args.filename, args.evaluate, args.visualize)

    best_weights_filename = "best_weights.txt"
    best_score_filename = "best_score.txt"

    if os.path.exists(best_weights_filename):
        print(f"Loading best weights from {best_weights_filename}")
        agent = SimpleCarAgent.from_file(best_weights_filename)
    else:
        agent = SimpleCarAgent()

    # Используем значение seed из аргумента --seed для обучения модели
    training_seed = args.seed
    w = load_world(training_seed, args.visualize)
    w.run(args.steps)

    agent_filename = f"agent_weights_seed_{training_seed}.txt"
    w.agents[0].to_file(agent_filename)

    seeds = [3, 13, 23]  # Список сидов для тестирования
    eval_scores = {}

    for seed in seeds:
        eval_score = test_agent(seed, agent_filename)
        eval_scores[seed] = eval_score
        print(f"Evaluation score for seed {seed}: {eval_score}")

    current_eval_score = test_agent(args.seed, agent_filename)
    print(f"Evaluation score for current seed {args.seed}: {current_eval_score}")

    print(f"Evaluation scores: {eval_scores}")
    avg_eval_score = np.mean(list(eval_scores.values()))
    print(f"Average evaluation score: {avg_eval_score}")

    if os.path.exists(best_score_filename):
        with open(best_score_filename, "r") as f:
            best_eval_scores = json.load(f)
    else:
        best_eval_scores = {}

    if not best_eval_scores or avg_eval_score > np.mean(list(best_eval_scores.values())):
        print(f"New best weights found with average score {avg_eval_score}")
        best_eval_scores = eval_scores
        w.agents[0].to_file(best_weights_filename)
        with open(best_score_filename, "w") as f:
            json.dump(best_eval_scores, f)
    else:
        print(f"Best weights unchanged with average score {np.mean(list(best_eval_scores.values()))}")

if __name__ == "__main__":
    main()
