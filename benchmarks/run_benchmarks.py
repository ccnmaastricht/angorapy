import json

import angorapy as ap
from angorapy.utilities.core import mpi_print

TASKS = [
    # DISCRETE
    "LunarLander-v2",
    "MountainCar-v0",
    "CartPole-v1",

    # CONTINUOUS
    "LunarLanderContinuous-v2",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v2",
    "Pendulum-v1",
    "Acrobot-v1",

    # ROBOTIC
    "Ant-v3",
    "Reacher-v2",
    "HalfCheetah-v2",
]

MODELS = [
    "simple",
    "wider",
    "deeper",
]

if __name__ == "__main__":
    NUM_REPEATS = 5

    results = {}

    total_runs = len(TASKS) * len(MODELS) * 2 * NUM_REPEATS
    current_run = 0
    mpi_print(f"Running benchmarks for {len(TASKS)} tasks and {len(MODELS)} models, totalling {total_runs} runs.")

    for task_name in TASKS:
        results[task_name] = {}
        for model_name in MODELS:
            results[task_name][model_name] = {}

            for model_type in ["ffn", "lstm"]:
                results[task_name][model_name][model_type] = []

                for _ in range(NUM_REPEATS):
                    current_run += 1
                    mpi_print(f"Running {task_name} - {model_name} - {model_type} [{current_run}/{total_runs}]")

                    task = ap.make_task(task_name)
                    model_builder = ap.get_model_builder(model=model_name, model_type=model_type, shared=False)

                    agent = ap.Agent(model_builder, task, workers=12, horizon=2048)
                    agent.drill(n=100, epochs=3, batch_size=512)

                    results[task_name][model_name][model_type].append(agent.cycle_reward_history)

                    with open("benchmark_results.json", "w") as f:
                        f.write(json.dumps(results, indent=4))
