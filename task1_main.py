import os
import glob
import torch
import argparse
import pandas as pd
import transformers

from rules_generator import get_python_rules
from task1_first_infer import task1_first_infer
from task1_refinement import task1_refinement
from evaluate_python import generate_full_validation_report, calculate_metrics


def evaluate_model(model, step, task, refinement, is_refinement):
    premises_dataset = "data/dataset_premises.csv"
    points_dataset = "data/points_dataset.csv"

    if refinement == "first":
        output_filename = f"outputs/Task1/{model}_first_outputs.csv"
    elif refinement == "syntax":
        output_filename = f"outputs/Task1/{model}_step_{step}_syntax_outputs.csv"
    elif refinement == "runtime":
        output_filename = f"outputs/Task1/{model}_step_{step}_runtime_outputs.csv"
    elif refinement == "logic":
        output_filename = f"outputs/Task1/{model}_step_{step}_logic_outputs.csv"

    outputs_dir = "saved_outputs/" + task + "_" + refinement + "_" + "outputs"
    # outputs_to_evaluate = glob.glob(f"outputs/{task_dir}/{outputs_format}")

    if not os.path.exists("saved_outputs"):
        os.mkdir("saved_outputs")

    if not os.path.exists("results"):
        os.mkdir("results") 

    if not os.path.exists(f"results/{refinement}"):
        os.mkdir(f"results/{refinement}")

    if not os.path.exists(f"results/{refinement}/{task}"):
        os.mkdir(f"results/{refinement}/{task}")

    if not os.path.exists("metrics"):
        os.mkdir("metrics")

    # for output_filename in outputs_to_evaluate:
    #     model_name = output_filename.split("/")[-1].split("_")[0]
    #     results_format = f"results/{refinement}/{task}/{model_name}_results.csv"
    #     generate_full_validation_report(is_refinement, task, model_name, output_filename, outputs_dir, premises_dataset, points_dataset, results_format)

    results_format = f"results/{refinement}/{task}/{model}_{refinement}_step_{step}_results.csv"
    generate_full_validation_report(is_refinement, task, model, output_filename, outputs_dir, premises_dataset, points_dataset, results_format)


def check_if_model_update(model, metrics, previous_metrics):
    current_metric_value = float(metrics[metrics["Model"] == model]["Overall"])
    previous_metric_value = float(previous_metrics[previous_metrics["Model"] == model]["Overall"])

    if current_metric_value == previous_metric_value:
        return False
    else:
        return True

models_to_eval = ["QwenCoderSmall"]

# First step of inference
# for model in models_to_eval:
#     task1_first_infer(model)
#     evaluate_model(model=model, task="Task1", step=0, refinement="first", is_refinement=False)

# calculate_metrics("results/first/Task1/*_first_step_0_results.csv", "metrics/Task1_first_metrics.csv")

# models_to_eval = ["Codestral"]

# start = 0
if_update = True
previous_refinement = "first"

# for step in [0]:
#     for refinement in ["logic"]:
#         for model in models_to_eval:
#             print(f"####################### {model}: {refinement} step {step}: INFERENCE #######################")
#             task1_refinement(model, refinement, step )
#             evaluate_model(model=model, task="Task1", step=step, refinement=refinement, is_refinement=True)

#         if len(glob.glob(f"results/{refinement}/Task1/*_{refinement}_step_{step}_results.csv")) > 0:
#             print(f"####################### {model}: {refinement} step {step}: METRICS CALCULATION #######################")
#             calculate_metrics(f"results/{refinement}/Task1/*_{refinement}_step_{step}_results.csv", f"metrics/Task1_{refinement}_step_{step}_metrics.csv")
#         else:
#             print(f"####################### {model}: {refinement} step {step}: NO NEW METRICS #######################")
#             break

start = 6
for step in range(start, 10):
    for refinement in ["syntax", "runtime", "logic"]:
        for model in models_to_eval:
            print(f"####################### {model}: {refinement} step {step}: INFERENCE #######################")
            task1_refinement(model, refinement, step )
            evaluate_model(model=model, task="Task1", step=step, refinement=refinement, is_refinement=True)

        if len(glob.glob(f"results/{refinement}/Task1/*_{refinement}_step_{step}_results.csv")) > 0:
            print(f"####################### {model}: {refinement} step {step}: METRICS CALCULATION #######################")
            calculate_metrics(f"results/{refinement}/Task1/*_{refinement}_step_{step}_results.csv", f"metrics/Task1_{refinement}_step_{step}_metrics.csv")
        else:
            print(f"####################### {model}: {refinement} step {step}: NO NEW METRICS #######################")
            break
        
        
# for step in range(start, 10):
#     for refinement in ["syntax", "runtime", "logic"]:
#         for model in models_to_eval:
#             if step > start:
#                 if_update = check_if_model_update(model, metrics, previous_metrics)
            
#             if not if_update:
#                 print(f"####################### {refinement} step {step}: NO UPDATE #######################")
#                 continue

#             print(f"####################### {refinement} step {step}: INFERENCE #######################")
#             task1_refinement(model, refinement, step )
#             evaluate_model(model=model, task="Task1", step=step, refinement=refinement, is_refinement=True)

#         if len(glob.glob(f"results/{refinement}/Task1/*_{refinement}_step_{step}_results.csv")) > 0:
#             print(f"####################### {refinement} step {step}: METRICS CALCULATION #######################")
#             calculate_metrics(f"results/{refinement}/Task1/*_{refinement}_step_{step}_results.csv", f"metrics/Task1_{refinement}_step_{step}_metrics.csv")
#         else:
#             print(f"####################### {refinement} step {step}: NO NEW METRICS #######################")
#             break

#         metrics = pd.read_csv(f"metrics/Task1_{refinement}_step_{step}_metrics.csv", delimiter=";", header=0)
        
#         if refinement != "syntax":
#             previous_metrics = pd.read_csv(f"metrics/Task1_{previous_refinement}_step_{step}_metrics.csv", delimiter=";", header=0)
#         elif step == 0 and refinement == "syntax":
#             previous_metrics = pd.read_csv(f"metrics/Task1_first_metrics.csv", delimiter=";", header=0)
#         elif step > 0 and refinement == "syntax":
#             previous_metrics = pd.read_csv(f"metrics/Task1_logic_step_{step-1}_metrics.csv", delimiter=";", header=0)

#         previous_refinement = refinement