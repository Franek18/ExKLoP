import os
import re
import ast
import glob
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_results_per_param():
    detail_results_dir = "detail_results"
    for result_type in ["first", "syntax", "runtime", "logic"]:
        for task in ["Task1", "Task2"]:
            for refinement_step in range(10):
                all_type_task_step_results = glob.glob(f"results/{result_type}/{task}/*step_{refinement_step}_results.csv")

                if not os.path.exists(f"results/{result_type}/{task}/{detail_results_dir}"):
                    os.mkdir(f"results/{result_type}/{task}/{detail_results_dir}")

                if refinement_step > 0:
                    syntax_report_df = pd.read_csv(f"results/{result_type}/{task}/{detail_results_dir}/{result_type}_{task}_step_{refinement_step-1}_detail_results_syntax.csv", delimiter=";", header=0)
                    final_report_df = pd.read_csv(f"results/{result_type}/{task}/{detail_results_dir}/{result_type}_{task}_step_{refinement_step-1}_detail_results.csv", delimiter=";", header=0)
                else:
                    if result_type == "first":
                        final_report_df = pd.DataFrame({"Parameters": [], "All_inputs": [], "Llama-70": [], "Llama-8": [], "Gemma3": [], "Codestral": [], "QwenCoder": [], "QwenCoderMedium": [], "QwenCoderSmall": []})
                        syntax_report_df = pd.DataFrame({"Parameters": [], "All_inputs": [], "Llama-70": [], "Llama-8": [], "Gemma3": [], "Codestral": [], "QwenCoder": [], "QwenCoderMedium": [], "QwenCoderSmall": []})
                    else:
                        syntax_report_df = pd.read_csv(f"results/first/{task}/{detail_results_dir}/first_{task}_step_0_detail_results_syntax.csv", delimiter=";", header=0)
                        final_report_df = pd.read_csv(f"results/first/{task}/{detail_results_dir}/first_{task}_step_0_detail_results.csv", delimiter=";", header=0)

                for results_csv in all_type_task_step_results:
                    csv_filename = results_csv.split("/")[-1]
                    model_name = csv_filename.split("_")[0]
                    results_df = pd.read_csv(results_csv, delimiter=";", header=0)

                    if task == "task2":
                        results_df = results_df[results_df["No. of parameters"] > 1]

                    final_report_df["Parameters"] = sorted(list(results_df["No. of parameters"].value_counts().to_dict().keys()))
                    syntax_report_df["Parameters"] = sorted(list(results_df["No. of parameters"].value_counts().to_dict().keys()))

                    # print(sorted(list(results_df["No. of parameters"].value_counts().to_dict().keys())))

                    syntax_res_df = results_df[results_df["Syntax eval"] == "Correct syntax"]
                    syntax_dict = syntax_res_df["No. of parameters"].value_counts().to_dict()
                    # syntax_per_param_results_dict = syntax_res_df["No. of parameters"].value_counts().to_dict()

                    answers_res_df = syntax_res_df[syntax_res_df["Outlier"] == syntax_res_df["Outlier detection"]]
                    answers_dict = answers_res_df["No. of parameters"].value_counts().to_dict()

                    for param in final_report_df["Parameters"]:
                        if param not in answers_dict.keys():
                            answers_dict[param] = 0
                        else:
                            answers_dict[param] = int(answers_dict[param])

                    syntax_per_param_results_df = pd.DataFrame({"No. of parameters": syntax_dict.keys(), "Counts": syntax_dict.values()})
                    syntax_per_param_results_df = syntax_per_param_results_df.sort_values(by=['No. of parameters'])
                    syntax_report_df[model_name] = syntax_per_param_results_df["Counts"].to_list()

                    answers_per_param_results_df = pd.DataFrame({"No. of parameters": answers_dict.keys(), "Counts": answers_dict.values()})
                    answers_per_param_results_df = answers_per_param_results_df.sort_values(by=['No. of parameters'])
                    if task == "task2":
                        print(f"{model_name}, {result_type}:\n", answers_per_param_results_df)
                        print(f"{model_name}, {result_type}:\n", answers_per_param_results_df["Counts"].to_list())
                    final_report_df[model_name] = answers_per_param_results_df["Counts"].to_list()
                
                all_inputs_arr = []
                for param in final_report_df["Parameters"]:
                    all_inputs_arr.append(results_df["No. of parameters"].value_counts().to_dict()[param])
                
                final_report_df["All_inputs"] = all_inputs_arr
                syntax_report_df["All_inputs"] = all_inputs_arr
                
                syntax_report_df.to_csv(f"results/{result_type}/{task}/{detail_results_dir}/{result_type}_{task}_step_{refinement_step}_detail_results_syntax.csv", sep=";", columns=list(syntax_report_df.keys()), index=False)
                final_report_df.to_csv(f"results/{result_type}/{task}/{detail_results_dir}/{result_type}_{task}_step_{refinement_step}_detail_results.csv", sep=";", columns=list(final_report_df.keys()), index=False)


def plot_LCR_changes_comp():
    all_task_results = sorted(glob.glob("results/*/*/detail_results/*results.csv"))

    all_results = {}

    for task_result in all_task_results:
        # print(task1_result)
        result_type = task_result.split("/")[2] + "_" + task_result.split("/")[1]
        task_result_df = pd.read_csv(task_result, delimiter=";", header=0)
        task_result_acc = {"Model name" : [], "OA metric value": [], "Parameter": []}
        # print(result_type)
        x = task_result_df["Parameters"].to_list()
        
        for model_name in task_result_df.columns[2:]:
            for _ in range(len(x)):
                task_result_acc["Model name"].append(model_name)
        
            FA = list(task_result_df[model_name].to_numpy() / task_result_df["All_inputs"].to_numpy())

            task_result_acc["OA metric value"].extend(FA)
            task_result_acc["Parameter"].extend(x)

        # for key in task_result_acc.keys():
        #     print(f"{key}: ", len(task_result_acc[key]))

        task1_result_acc_df = pd.DataFrame(task_result_acc)
        all_results[result_type] = task1_result_acc_df

    desc_fontsize = 50

    custom_palette = ["blue", "orange", "green", "red", "magenta", "lightblue"]

    titles = [['Task1\ninference', 'Task 1\nsyntax correction', 'Task 1\nprogram correction', 'Task 1\nlogic correction'],
              ['Task2\ninference', 'Task 2\nsyntax correction', 'Task 2\nprogram correction', 'Task 2\nlogic correction']]

    topics = {'Task1\ninference': "task1_first_iter", 'Task 1\nsyntax correction': "task1_syntax", 'Task 1\nprogram correction': "task1_runtime",
              'Task 1\nlogic correction': "task1_critic_rules", 'Task2\ninference': "task2_first_iter", 'Task 2\nsyntax correction': "task2_syntax",
              'Task 2\nprogram correction': "task2_runtime", 'Task 2\nlogic correction': "task2_critic_rules"}
    
    fig, axes = plt.subplots(2, 4, figsize=(40, 30), sharey=True)
    # print(axes)
    for row_axes, row_titles in zip(axes, titles):
        for ax, title in zip(row_axes, row_titles):
            # print(all_results[title])
            topic = topics[title]
            sns.pointplot(ax=ax, data=all_results[topic], x="Parameter", y="OA metric value", hue="Model name", dodge=True, palette=custom_palette)

            ax.set_ylim([0.0, 1.1])     
            ax.tick_params(axis='x', labelsize=desc_fontsize*0.8)
            ax.tick_params(axis='y', labelsize=desc_fontsize*0.8)
            ax.set_title(title, fontsize=desc_fontsize)
            ax.legend_.remove()
            ax.set_xlabel("")
            ax.set_ylabel("")
    
    handles, labels = axes[1][3].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=desc_fontsize*1.2)

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0.05, 0.2, 1, 1])            

    # naming the x axis
    fig.supylabel(f'Logical Consistency Rate', x=0.01, y=0.65, fontsize=desc_fontsize*1.2)
    
    plt.savefig(f"images/png/LCR_changes_comp.png")
    plt.savefig(f"images/pdf/LCR_changes_comp.pdf")


def plot_FSR_changes_comp():
    all_task_results = sorted(glob.glob("results/*/*/detail_results/*syntax.csv"))
    # task2_all_results = sorted(glob.glob(f"results/*/task2/detail_results/*syntax.csv"))

    all_results = {}

    for task_result in all_task_results:
        # print(task1_result)
        result_type = task_result.split("/")[2] + "_" + task_result.split("/")[1]
        task_result_df = pd.read_csv(task_result, delimiter=";", header=0)
        task_result_acc = {"Model name" : [], "FSR metric value": [], "Parameter": []}
        # print(result_type)
        x = task_result_df["Parameters"].to_list()
        
        for model_name in task_result_df.columns[2:]:
            for _ in range(len(x)):
                task_result_acc["Model name"].append(model_name)
        
            FA = list(task_result_df[model_name].to_numpy() / task_result_df["All_inputs"].to_numpy())

            task_result_acc["FSR metric value"].extend(FA)
            task_result_acc["Parameter"].extend(x)

        # for key in task_result_acc.keys():
        #     print(f"{key}: ", len(task_result_acc[key]))

        task1_result_acc_df = pd.DataFrame(task_result_acc)
        all_results[result_type] = task1_result_acc_df

    desc_fontsize = 50

    custom_palette = ["blue", "orange", "green", "red", "magenta", "lightblue"]

    titles = [['Task1\ninference', 'Task 1\nsyntax correction', 'Task 1\nprogram correction', 'Task 1\nlogic correction'],
              ['Task2\ninference', 'Task 2\nsyntax correction', 'Task 2\nprogram correction', 'Task 2\nlogic correction']]

    topics = {'Task1\ninference': "task1_first_iter", 'Task 1\nsyntax correction': "task1_syntax", 'Task 1\nprogram correction': "task1_runtime",
              'Task 1\nlogic correction': "task1_critic_rules", 'Task2\ninference': "task2_first_iter", 'Task 2\nsyntax correction': "task2_syntax",
              'Task 2\nprogram correction': "task2_runtime", 'Task 2\nlogic correction': "task2_critic_rules"}

    ############ TASK 1 ############
    fig, axes = plt.subplots(2, 4, figsize=(40, 30), sharey=True)
    # print(axes)
    for row_axes, row_titles in zip(axes, titles):
        for ax, title in zip(row_axes, row_titles):
            # print(all_results[title])
            topic = topics[title]
            sns.pointplot(ax=ax, data=all_results[topic], x="Parameter", y="FSR metric value", hue="Model name", dodge=True, palette=custom_palette)

            ax.set_ylim([0.0, 1.1])     
            ax.tick_params(axis='x', labelsize=desc_fontsize*0.8)
            ax.tick_params(axis='y', labelsize=desc_fontsize*0.8)
            ax.set_title(title, fontsize=desc_fontsize)
            ax.legend_.remove()
            ax.set_xlabel("")
            ax.set_ylabel("")
    
    handles, labels = axes[1][3].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=desc_fontsize*1.2)

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0.05, 0.2, 1, 1])            

    # naming the x axis
    fig.supylabel(f'Formalization Success Rate', x=0.01, y=0.65, fontsize=desc_fontsize*1.2)
    
    plt.savefig(f"images/png/FSR_changes_comp.png")
    plt.savefig(f"images/pdf/FSR_changes_comp.pdf")
    

def seaborn_plot_line_oa_metric_changes():
    all_inputs = 50

    task1_metrics_files = sorted(glob.glob("metrics/Task1_logic_step_*_metrics.csv").append("metrics/Task1_first_metrics.csv"))
    task2_metrics_files = sorted(glob.glob("metrics/Task2_logic_step_*_metrics.csv").append("metrics/Task2_first_metrics.csv"))

    task1_results_df = {"Model name" : [], "LCR metric value": [], "Step": []}
    task2_results_df = {"Model name" : [], "LCR metric value": [], "Step": []}

    # steps = {"first_iter": "First\nstep", "critic_syntax": "Syntax\ncorrection", "critic_runtime": "Code\ncorrection", "critic_rules": "Logic\ncorrection"}

    steps = list(range(1,11))

    for task_metric in task1_metrics_files:
        # result_type = task_metric.split("/")[-1].split("_")[4] + "_" + task_metric.split("/")[-1].split("_")[5]
        step = int(task_metric.split("/")[-1].split("_")[-2]) + 1
        task_metric_df = pd.read_csv(task_metric, delimiter=";", header=0)

        for model_name in task_metric_df["Model"]:
            # print(model_name)
            result_idx = task_metric_df[task_metric_df["Model"] == model_name].index[0]
            oa_metric_result = task_metric_df["Overall"][result_idx]
            # print(oa_metric_result)
            task1_results_df["LCR metric value"].append(oa_metric_result)
            task1_results_df["Model name"].append(model_name)
            task1_results_df["Step"].append(step)


    for task_metric in task2_metrics_files:
        step = int(task_metric.split("/")[-1].split("_")[-2]) + 1
        task_metric_df = pd.read_csv(task_metric, delimiter=";", header=0)

        for model_name in task_metric_df["Model"]:
            result_idx = task_metric_df[task_metric_df["Model"] == model_name].index[0]
            oa_metric_result = task_metric_df["Overall"][result_idx]
            # print(oa_metric_result)
            task2_results_df["LCR metric value"].append(oa_metric_result)
            task2_results_df["Model name"].append(model_name)
            task2_results_df["Step"].append(step)

    task1_results_df = pd.DataFrame(task1_results_df)
    task2_results_df = pd.DataFrame(task2_results_df)
    final_dataset_df = pd.concat([task1_results_df, task2_results_df])

    # print(task1_results_df)
    # sns.set()

    desc_fontsize = 60
    fig, axes = plt.subplots(1, 2, figsize=(40, 20), sharey=True)
    # fig.suptitle('Overall Accuracy changes')
    # axes[0].set_title('Task 1 Overall Accuracy changes', fontweight ='bold', fontsize=desc_fontsize*1.2)
    # axes[1].set_title('Task 2 Overall Accuracy changes', fontweight ='bold', fontsize=desc_fontsize*1.2)
    
    axes[0].tick_params(axis='x', labelsize=desc_fontsize)
    axes[0].tick_params(axis='y', labelsize=desc_fontsize)
    axes[1].tick_params(axis='x', labelsize=desc_fontsize)
    axes[1].tick_params(axis='y', labelsize=desc_fontsize)
    axes[0].set_ylim(0, 1.1)
    axes[1].set_ylim(0, 1.1)
    axes[0].set_xlim(-0.5, len(task1_results_df['Step'].unique()) - 0.5)
    axes[1].set_xlim(-0.5, len(task2_results_df['Step'].unique()) - 0.5)

    custom_palette = ["blue", "orange", "green", "red", "magenta", "purple", "lightblue"]

    sns.pointplot(ax=axes[0], data=task1_results_df, x="Step", y="LCR metric value", markersize=32, hue="Model name", dodge=True, order=["First\nstep", "Syntax\ncorrection", "Code\ncorrection", "Logic\ncorrection"], palette=custom_palette)
    sns.pointplot(ax=axes[1], data=task2_results_df, x="Step", y="LCR metric value", markersize=32, hue="Model name", dodge=True, order=["First\nstep", "Syntax\ncorrection", "Code\ncorrection", "Logic\ncorrection"], palette=custom_palette)
    
    # wrap_labels(axes[0])
    # wrap_labels(axes[1])
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=desc_fontsize)

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")

    axes[0].set_title("Task 1. Range Checking", fontsize=desc_fontsize*1.2)
    axes[1].set_title("Task 2. Constraint Validation", fontsize=desc_fontsize*1.2)

    # fig.supxlabel('Steps of inference', fontweight ='bold', fontsize=desc_fontsize*1.2)
    fig.supylabel(f'Logical Consistency Rate', x=0.005, y=0.65, fontsize=desc_fontsize*1.2)

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    plt.savefig(f"images/pdf/oa_metric_changes_through_steps.pdf")
    plt.savefig(f"images/png/oa_metric_changes_through_steps.png")


def plot_OA_before_after_comp():
    task1_before_results_df = pd.read_csv(f"results/first/Task1/detail_results/first_Task1_step_0_detail_results.csv", delimiter=";", header=0)
    task1_after_results_df = pd.read_csv(f"results/logic/Task1/detail_results/logic_Task1_step_9_detail_results.csv", delimiter=";", header=0)

    task2_before_results_df = pd.read_csv(f"results/first/Task2/detail_results/first_Task2_step_0_detail_results.csv", delimiter=";", header=0)
    task2_after_results_df = pd.read_csv(f"results/logic/Task2/detail_results/logic_Task2_step_9_detail_results.csv", delimiter=";", header=0)

    x1 = task1_before_results_df["Parameters"].to_list()
    x2 = task2_before_results_df["Parameters"].to_list()

    labels = list(task1_before_results_df.columns[2:])   

    Acc_before_task1 = {"Model name" : [], "LCR metric value": [], "Parameter": []}
    Acc_after_task1 = {"Model name" : [], "LCR metric value": [], "Parameter": []}

    Acc_before_task2 = {"Model name" : [], "LCR metric value": [], "Parameter": []}
    Acc_after_task2 = {"Model name" : [], "LCR metric value": [], "Parameter": []}

    for model_name in task1_before_results_df.columns[2:]:
        for _ in range(len(x1)):
            Acc_before_task1["Model name"].append(model_name)
            Acc_after_task1["Model name"].append(model_name)
        
        acc_before = list(task1_before_results_df[model_name].to_numpy() / task1_before_results_df["All_inputs"].to_numpy())
        acc_after = list(task1_after_results_df[model_name].to_numpy() / task1_after_results_df["All_inputs"].to_numpy())

        Acc_before_task1["LCR metric value"].extend(acc_before)
        Acc_after_task1["LCR metric value"].extend(acc_after)

        Acc_before_task1["Parameter"].extend(x1)
        Acc_after_task1["Parameter"].extend(x1)        

    for model_name in task2_before_results_df.columns[2:]:
        for _ in range(len(x2)):
            Acc_before_task2["Model name"].append(model_name)
            Acc_after_task2["Model name"].append(model_name)
        
        acc_before = list(task2_before_results_df[model_name].to_numpy() / task2_before_results_df["All_inputs"].to_numpy())
        acc_after = list(task2_after_results_df[model_name].to_numpy() / task2_after_results_df["All_inputs"].to_numpy())

        Acc_before_task2["LCR metric value"].extend(acc_before)
        Acc_after_task2["LCR metric value"].extend(acc_after)

        Acc_before_task2["Parameter"].extend(x2)
        Acc_after_task2["Parameter"].extend(x2)    

    # print(Acc_before_task1)
    Acc_before_task1_df = pd.DataFrame(Acc_before_task1)
    Acc_after_task1_df = pd.DataFrame(Acc_after_task1)

    Acc_before_task2_df = pd.DataFrame(Acc_before_task2)
    Acc_after_task2_df = pd.DataFrame(Acc_after_task2)

    desc_fontsize = 72

    custom_palette = ["blue", "orange", "green", "red", "magenta", "purple", "lightblue"]

    ############ TASK 1 ############
    fig, axes = plt.subplots(1, 2, figsize=(36, 24), sharey=True)

    x_min, x_max = int(x1[0]), int(x1[-1])

    models = Acc_after_task1_df["Model name"].drop_duplicates().to_list()

    for ax, results, title in zip(axes, [Acc_before_task1_df, Acc_after_task1_df], ["Task 1. Range Checking\ninitial LCR", "Task 1. Range Checking\nfinal LCR"]):
        for model_name, color in zip(models, custom_palette):
            subset = results[results['Model name'] == model_name]
            sns.regplot(ax=ax, x="Parameter", y="LCR metric value", ci=None, data=subset, scatter_kws={'color': color, 's': 256}, label=model_name, line_kws={'color': color, 'markersize': 128})
        
        ax.set_title(title, fontsize=desc_fontsize)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([0.0, 1.1])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=desc_fontsize)
        ax.tick_params(axis='y', labelsize=desc_fontsize)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, markerscale=3.0, frameon=False, fontsize=0.8*desc_fontsize)
    fig.supylabel(f'Logical Consistency Rate', x=0.03, y=0.60, fontsize=desc_fontsize)
    fig.supxlabel('Number of parameters to translate', y=0.2, fontsize=desc_fontsize)


    plt.tight_layout(rect=[0.05, 0.25, 1, 0.9])

    plt.savefig(f"images/png/task1_OA_before_after_comp.png")
    plt.savefig(f"images/pdf/task1_OA_before_after_comp.pdf")

    ############ TASK 2 ############
    fig, axes = plt.subplots(1, 2, figsize=(36, 24), sharey=True)
    models = Acc_after_task2_df["Model name"].drop_duplicates().to_list()

    x_min, x_max = int(x2[0]), int(x2[-1])

    for ax, results, title in zip(axes, [Acc_before_task2_df, Acc_after_task2_df], ["Task 2. Constraint Validation\ninitial LCR", "Task 2. Constraint Validation\nfinal LCR"]):
        for model_name, color in zip(models, custom_palette):
            subset = results[results['Model name'] == model_name]
            sns.regplot(ax=ax, x="Parameter", y="LCR metric value", ci=None, data=subset, scatter_kws={'color': color, 's': 256}, label=model_name, line_kws={'color': color, 'markersize': 128})
        
        ax.set_title(title, fontsize=desc_fontsize)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([0.0, 1.1])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=desc_fontsize)
        ax.tick_params(axis='y', labelsize=desc_fontsize)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, markerscale=2.0, frameon=False, fontsize=0.8*desc_fontsize)
    fig.supylabel(f'Logical Consistency Rate', x=0.03, y=0.60, fontsize=desc_fontsize)
    fig.supxlabel('Number of relations to translate', y=0.2, fontsize=desc_fontsize)


    plt.tight_layout(rect=[0.05, 0.25, 1, 0.9])

    plt.savefig(f"images/png/task2_OA_before_after_comp.png")
    plt.savefig(f"images/pdf/task2_OA_before_after_comp.pdf")


def replace_outlier_with_error():
    task1_results = glob.glob("results/*/Task1/*.csv")
    task2_results = glob.glob("results/*/Task2/*.csv")

    pattern = r"""^\[(?:'(?:\s*(?:True|False)\s*(?:,\s*(?:True|False)\s*)*)?'(?:,\s*'(?:\s*(?:True|False)\s*(?:,\s*(?:True|False)\s*)*)?')*)?\]$"""


    for task1_result, task2_result in zip(task1_results, task2_results):
        task1_df = pd.read_csv(task1_result, delimiter=";", header=0)
        task2_df = pd.read_csv(task2_result, delimiter=";", header=0)

        task1_df['Outlier'] =task1_df['Outlier'].apply(lambda x: x if re.fullmatch(pattern, x, re.VERBOSE) else 'Error')
        task2_df['Outlier'] = task2_df['Outlier'].apply(lambda x: x if re.fullmatch(pattern, x, re.VERBOSE) else 'Error')

        task1_df.to_csv(task1_result, sep=";", columns=list(task1_df.keys()), index=False)
        task2_df.to_csv(task2_result, sep=";", columns=list(task2_df.keys()), index=False)


def merge_refinement_metrics():
    task1_metrics = sorted(glob.glob("metrics/Task1_logic_step_*_metrics.csv"))
    task2_metrics = sorted(glob.glob("metrics/Task2_logic_step_*_metrics.csv"))

    preview_task1_final_metrics = {}
    preview_task2_final_metrics = {}

    task1_final_metrics = {"Model": [], "Formalization": [], "Overall": [], "Step": []}
    task2_final_metrics = {"Model": [], "Formalization": [], "Overall": [], "Step": []}

    task1_models = []
    task2_models = []

    for task1_metric_filename in task1_metrics:
        task1_metric_df = pd.read_csv(task1_metric_filename, delimiter=";", header=0)
        refinement_step = task1_metric_filename.split("/")[-1].split("_")[3]
        

        for idx in task1_metric_df.index:
            model_name = task1_metric_df["Model"][idx]
            
            if model_name not in task1_models:
                task1_models.append(model_name)

            
            if model_name + "_Overall" not in preview_task1_final_metrics.keys():
                print("Task1 current key: ", model_name + "_Overall")
                print("Task1 keys: ", preview_task1_final_metrics.keys())
                print("Task1 refinement_step = ", refinement_step)

                preview_task1_final_metrics[model_name + "_step"] = int(refinement_step) + 1
                preview_task1_final_metrics[model_name + "_Formalization"] = task1_metric_df["Formalization"][idx]
                preview_task1_final_metrics[model_name + "_Overall"] = task1_metric_df["Overall"][idx]

            elif model_name + "_Overall" in preview_task1_final_metrics.keys() and float(task1_metric_df["Overall"][idx]) > float(preview_task1_final_metrics[model_name + "_Overall"]):
                new_value = task1_metric_df["Overall"][idx]
                current_value = preview_task1_final_metrics[model_name + "_Overall"]
                print("Task1 refinement_step = ", refinement_step)

                print(f"Task1 {model_name} + _Overall value =  {new_value}")
                print(f"Task1 preview value: {current_value}")
                
                preview_task1_final_metrics[model_name + "_step"] = int(refinement_step) + 1
                preview_task1_final_metrics[model_name + "_Formalization"] = task1_metric_df["Formalization"][idx]
                preview_task1_final_metrics[model_name + "_Overall"] = task1_metric_df["Overall"][idx]

    for task2_metric_filename in task2_metrics:
        task2_metric_df = pd.read_csv(task2_metric_filename, delimiter=";", header=0)
        refinement_step = task2_metric_filename.split("/")[-1].split("_")[3]

        for idx in task2_metric_df.index:
            model_name = task2_metric_df["Model"][idx]

            if model_name not in task2_models:
                task2_models.append(model_name)
         
            if model_name + "_Overall" not in preview_task2_final_metrics.keys():
                # print("Task2 current key: ", model_name + "_Overall")
                # print("Task2 keys: ", preview_task2_final_metrics.keys())

                preview_task2_final_metrics[model_name + "_step"] = int(refinement_step) + 1
                preview_task2_final_metrics[model_name + "_Formalization"] = task2_metric_df["Formalization"][idx]
                preview_task2_final_metrics[model_name + "_Overall"] = task2_metric_df["Overall"][idx]

            elif model_name + "_Overall" in preview_task2_final_metrics.keys() and float(task2_metric_df["Overall"][idx]) > float(preview_task2_final_metrics[model_name + "_Overall"]):
                new_value = task2_metric_df["Overall"][idx]
                current_value = preview_task2_final_metrics[model_name + "_Overall"]

                # print(f"Task2 {model_name} + _Overall value =  {new_value}")
                # print(f"Task2 preview value: {current_value}")

                preview_task2_final_metrics[model_name + "_step"] = int(refinement_step) + 1
                preview_task2_final_metrics[model_name + "_Formalization"] = task2_metric_df["Formalization"][idx]
                preview_task2_final_metrics[model_name + "_Overall"] = task2_metric_df["Overall"][idx]


    # print("Task1 preview:\n", preview_task1_final_metrics)
    # print("Task2 preview:\n", preview_task2_final_metrics)

    for task1_model_name in task1_models:
        task1_final_metrics["Model"].append(task1_model_name)
        task1_final_metrics["Formalization"].append(preview_task1_final_metrics[task1_model_name + "_Formalization"])
        task1_final_metrics["Overall"].append(preview_task1_final_metrics[task1_model_name + "_Overall"])
        task1_final_metrics["Step"].append(preview_task1_final_metrics[task1_model_name + "_step"])


    for task2_model_name in task2_models:
        task2_final_metrics["Model"].append(task2_model_name)
        task2_final_metrics["Formalization"].append(preview_task2_final_metrics[task2_model_name + "_Formalization"])
        task2_final_metrics["Overall"].append(preview_task2_final_metrics[task2_model_name + "_Overall"])
        task2_final_metrics["Step"].append(preview_task2_final_metrics[task2_model_name + "_step"])


    task1_final_metrics_df = pd.DataFrame(task1_final_metrics)
    task2_final_metrics_df = pd.DataFrame(task2_final_metrics)

    task1_final_metrics_df = task1_final_metrics_df.sort_values(by=["Overall"], ascending=False)
    task1_final_metrics_df.to_csv("metrics/Task1_final_refinement_metrics.csv", sep=";", columns=list(task1_final_metrics_df.keys()), index=False)

    task2_final_metrics_df = task2_final_metrics_df.sort_values(by=["Overall"], ascending=False)
    task2_final_metrics_df.to_csv("metrics/Task2_final_refinement_metrics.csv", sep=";", columns=list(task2_final_metrics_df.keys()), index=False)


def update_refinement_metrics():
    
# merge_refinement_metrics()
# replace_outlier_with_error()
# save_results_per_param()
# plot_OA_before_after_comp()
seaborn_plot_line_oa_metric_changes()
# plot_FSR_changes_comp()
# plot_LCR_changes_comp()
