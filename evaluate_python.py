import os
import re
import ast
import glob
import subprocess
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score


def generate_final_rule_task1(prompt_params):
    # Main function should return answer argregated for all rules, and for each rule separately
    # return not(r1(arg1) and r2(arg2)), r1(arg1), r2(arg2)
    prompt_no_params = len(prompt_params)
    final_rule = f"def r{prompt_no_params + 1}([args]):\n    return not([call])[all_calls]"
    prompt_no_params = 0

    final_rule_args = ""
    final_rule_function = ""
    all_calls = ""
    arg_idx = 1
    rule_idx = 1
    for params in prompt_params:
        prompt_no_params += len(params)
        final_rule_function += f"r{rule_idx}("
        all_calls += f", r{rule_idx}("

        final_rule_args += f"arg{arg_idx}: float, "
        final_rule_function += f"arg{arg_idx}"
        all_calls += f"arg{arg_idx}"
        arg_idx += 1
      
        final_rule_function += ") and "
        all_calls = all_calls + ")"

        rule_idx += 1

    final_rule_args = final_rule_args[:-2]
    final_rule_function = final_rule_function[:-5]

    final_rule = final_rule.replace("[args]", final_rule_args).replace("[call]", final_rule_function).replace("[all_calls]", all_calls)

    return final_rule


def generate_final_rule_task2(parameters_df, idx):
    # Main function should return answer argregated for all rules, and for each rule separately
    # return not(r1(arg1, arg2) and r2(arg3, arg4)), r1(arg1, arg2), r2(arg3, arg4)
    prompt_no_conditions = parameters_df["Number of conditions"][idx]
    # print(f"Prompt no. {idx} - number of conditions: {prompt_no_conditions}")
    final_rule = f"def r{prompt_no_conditions + 1}([args]):\n    return not([call])[all_calls]"
    prompt_params = ast.literal_eval(parameters_df["Parameters"][idx])
    # print("Prompt params: ", prompt_params)
    prompt_no_params = 0

    final_rule_args = ""
    final_rule_function = ""
    all_calls = ""
    arg_idx = 1
    rule_idx = 1
    for cond_params in prompt_params:
        prompt_no_params += len(cond_params)
        final_rule_function += f"r{rule_idx}("
        all_calls += f", r{rule_idx}("
        for _ in range(len(cond_params)):
            final_rule_args += f"arg{arg_idx}: float, "
            final_rule_function += f"arg{arg_idx}, "
            all_calls += f"arg{arg_idx}, "
            arg_idx += 1
      
        final_rule_function = final_rule_function[:-2]
        final_rule_function += ") and "
        all_calls = all_calls[:-2] + ")"

        rule_idx += 1

    final_rule_args = final_rule_args[:-2]
    final_rule_function = final_rule_function[:-5]

    final_rule = final_rule.replace("[args]", final_rule_args).replace("[call]", final_rule_function).replace("[all_calls]", all_calls)

    # print(f"Final rule for prompt {idx}: {final_rule}")
    return final_rule

def evaluate_outputs_syntax(task, is_revised, model_name, outputs_df, report_dict, premises_dataset, points_dataset, system_prompt):
    # Load a csv file with outputs from LLMs
    points_dataset_df = pd.read_csv(points_dataset, delimiter=";", header=0)
    premises_dataset_df = pd.read_csv(premises_dataset, delimiter=";", header=0)
    # conditions_df = pd.read_csv("data/Task2_conditions_updated.csv", delimiter=";", header=0)

    if is_revised:
        for idx in tqdm(outputs_df.index):
            python_output = outputs_df["Model output"][idx]

            if task == "Task2":
                all_rules = re.findall(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*:\n(?: {4}.+\n?)+", python_output)
                # TODO any changes needed here?
                final_rule = generate_final_rule_task2(premises_dataset_df[["Number of conditions", "Parameters"]], idx)
                # We must get indices of premises inside the prompt
            else:
                prompt_params = ast.literal_eval(outputs_df["Parameters"][idx])
                final_rule = generate_final_rule_task1(prompt_params)
                all_rules = re.findall(r"def .+?:\n\s+return .+", python_output)

            all_comments = re.findall(r"(#.*\n)", python_output)
        
            if len(all_comments) != len(all_rules):
                python_code = "\n".join(all_rules)
            else:
                python_code = ""
                for comment, rule in zip(all_comments, all_rules):
                    python_code += comment + rule + "\n"

            
            # Add final rule to the code
            python_code += "\n" + final_rule
            report_dict["Model output"][idx] = python_code

    else:
        for idx in tqdm(outputs_df.index):
            used_params = []
            input_prompt = outputs_df["Prompt"][idx]
            input_premises = input_prompt.split("Textual context: ")[-1]
            # full_prompt = system_prompt + "\n" + input_prompt
            full_prompt = input_prompt

            report_dict["Prompt"].append(full_prompt)
            report_dict["Premises"].append(input_premises)

            is_outlier = False

            python_output = outputs_df[model_name][idx]

            if task == "Task2":
                all_rules = re.findall(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*:\n(?: {4}.+\n?)+", python_output)
                # We must get indices of premises inside the prompt
                no_used_conditions = premises_dataset_df["Number of conditions"][idx]
                # used_conditions = outputs_df["Prompt"][idx].split("Textual context: ")[-1].split("\n")[:-1]
                # no_used_conditions = len(used_conditions)
                report_dict["No. of parameters"].append(no_used_conditions)
                report_dict["Parameters"].append(premises_dataset_df["Parameters"][idx])

                # Genereta a final rule for a python code
                final_rule = generate_final_rule_task2(premises_dataset_df[["Number of conditions", "Parameters"]], idx)
            else:           
                all_rules = re.findall(r"def .+?:\n\s+return .+", python_output)
                parameters = list(premises_dataset_df.columns[2:])
                for parameter in parameters:
                    parameter_indicator = int(premises_dataset_df[parameter][idx])
                    if parameter_indicator == 1:
                        used_params.append(parameter)

                report_dict["No. of parameters"].append(len(used_params))
                report_dict["Parameters"].append(used_params)

                # Genereta a final rule for a python code
                final_rule = generate_final_rule_task1(used_params)

                        # parameter_outliers = points_dataset_df[points_dataset_df["Parameter"] == parameter]["Outlier"].to_list()
                        # if sum(parameter_outliers) > 0:
                        #     is_outlier = True
                            # break       
                # if is_outlier:
                #     report_dict["Outlier"].append(True)
                #     # By default we negate Outlier detection which will be change in second evaluation step
                #     report_dict["Outlier detection"].append(False)                
                # elif not is_outlier:
                #     report_dict["Outlier"].append(False)
                #     # By default we negate Outlier detection which will be change in second evaluation step
                #     report_dict["Outlier detection"].append(True)

            report_dict["Outlier"].append("Error")
            # By default we negate Outlier detection which will be change in second evaluation step
            report_dict["Outlier detection"].append("Error")  
            

            all_comments = re.findall(r"(#.*\n)", python_output)
            
            if len(all_comments) != len(all_rules):
                python_code = "\n".join(all_rules)
            else:
                python_code = ""
                for comment, rule in zip(all_comments, all_rules):
                    python_code += comment + rule + "\n"

            # Add final rule to the code
            python_code += "\n" + final_rule

            report_dict["Model"].append(model_name)
            report_dict["Model output"].append(python_code)
            f = open("output_python.py", "w")
            f.write(python_code)
            f.close()

            # output = os.popen('python output_python.py').read()
            output = subprocess.run(['python', 'output_python.py'], capture_output=True, text=True).stderr
            if output == '':
                report_dict["Syntax eval"].append("Correct syntax")
            else:
                report_dict["Syntax eval"].append(output)


def save_outputs_task1(model_name, report_dict, dataset_filename, points_filename, output_dir, samples_per_param=50):
    # Load results of syntax evaluation
    outputs_df = pd.DataFrame({"Model output": report_dict["Model output"], "Syntax eval": report_dict["Syntax eval"]})
    dataset_df = pd.read_csv(dataset_filename, delimiter=";", header=0)
    datapoints_df = pd.read_csv(points_filename, delimiter=";", header=0)
    parameters = dataset_df.columns[2:]

    columns = outputs_df.columns
    range_of_parameters = dataset_df["Number of parameters"].unique()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(output_dir + "/" + model_name):
        os.mkdir(output_dir + "/" + model_name)

    idx = 0
    for no_of_parameters in range_of_parameters:
        lean_code_path = output_dir + "/" + model_name + "/" + str(no_of_parameters) + "/python_code"
        datapoints_path = output_dir + "/" + model_name + "/" + str(no_of_parameters) + "/datapoints"

        if not os.path.exists(output_dir + "/" + model_name + "/" + str(no_of_parameters)):
            os.mkdir(output_dir + "/" + model_name + "/" + str(no_of_parameters))

        if not os.path.exists(lean_code_path):
            os.mkdir(lean_code_path)

        if not os.path.exists(datapoints_path):
            os.mkdir(datapoints_path)

            # samples_per_param = dataset_df["Number of parameters"].value_counts()[no_of_parameters]

        param_outputs_df = outputs_df[idx:idx+samples_per_param]
        idx += samples_per_param
        for idx2 in param_outputs_df[param_outputs_df["Syntax eval"] == "Correct syntax"].index:
            input_datapoints = {}
            # Check which parameters are present in the prompt
            for parameter in parameters:
                if dataset_df[parameter][idx2] == 1:
                    # And retrieve all datapoints for these parameters
                    input_datapoints[parameter + " value"] = [datapoints_df["Value"][idx3] for idx3 in datapoints_df[datapoints_df["Parameter"] == parameter].index]
                    input_datapoints[parameter + " outlier"] = [datapoints_df["Outlier"][idx3] for idx3 in datapoints_df[datapoints_df["Parameter"] == parameter].index]
                
            # Get a lean code generate by a model
            # python_code = outputs_df["Model output"][idx2] + "\n"

            python_output = outputs_df["Model output"][idx2] + "\n"
            all_comments = re.findall(r"(#.*\n)", python_output)
            rules = re.findall(r"def .+?:\n\s+return .+", python_output)

            if len(all_comments) != len(rules):
                python_code = "\n".join(rules)
            else:
                python_code = ""
                for comment, rule in zip(all_comments, rules):
                    python_code += comment + rule + "\n"

            # Retrieve all rules generated by a model
            rules = np.unique(re.findall("r[0-9]+", python_code))

            # The output can be a valid lean code but without any rules defined ...
            if len(rules) == 0:
                # correct_rules = False
                 continue

            # With number of rules > 9 we need to solve the problem with the order of decimal rules r10, r11 ... in a sorted list
            if len(rules) > 9:
                new_rules = [rules[0]]
                rules_len = len(rules)
                decimal_no = rules_len - 9
                decimals = rules[1:decimal_no + 1]
                for rule in rules[decimal_no+1:]:
                    new_rules.append(rule)

                new_rules.extend(decimals)
                rules = new_rules

            # Create a lean line of code for evaluation of rules
            # by evaluating the last, general, rule
            # eval_line = f"\nprint({rules[-1]}([args]))"

            # Pass arguments to the general rule, first other rules
            # for j in range(len(rules)-1):
            #     eval_line += rules[j] + " "

            # Next datapoints for these rules
            # no_datapoints = len(list(input_datapoints.values())[0])
            input_parameters = list(input_datapoints.keys())
            no_input_parameters = len(list(input_datapoints.keys()))

            # datapoints for each parameter = [outlier_low, no-outlier_low, no-outlier_high, outlier_high]

            for j in [0, 3]: # indices for outliers
                for m in [1, 2]: # indices for no-outliers
                    for k in range(0, no_input_parameters-1, 2):
                        eval_line = f"\nprint({rules[-1]}([args]))"
                        eval_arguments = ""
                        for l in range(0, no_input_parameters-1, 2):
                            if k == l: # Check which param should be testified for outliers
                                # Set an outlier data point for this parameter
                                eval_arguments += str(input_datapoints[input_parameters[l]][j]) + ", "
                            else:
                                # Set a no-outlier data point for this parameter
                                eval_arguments += str(input_datapoints[input_parameters[l]][m]) + ", "
                    
                        # This is the one example of rules verification
                        python_code += eval_line.replace("[args]", eval_arguments[:-2]) + "\n"
            
        #     print(lean_code)
        #     break
        # break
            input_datapoints_df = pd.DataFrame(input_datapoints)
            input_datapoints_df.to_csv(datapoints_path  + "/" + str(idx2) + "_datapoints.csv", sep=";", columns=list(input_datapoints_df.keys()), index=False)

            f = open(lean_code_path + "/" + str(idx2) + "_output_python.py", "w")
            f.write(python_code)
            f.close()


def save_outputs_task2(model_name, report_dict, dataset_filename, points_filename, output_dir, samples_per_param=50):
    # Load results of syntax evaluation
    outputs_df = pd.DataFrame({"Model output": report_dict["Model output"], "Number of conditions": report_dict["No. of parameters"], "Syntax eval": report_dict["Syntax eval"]})
    dataset_df = pd.read_csv(dataset_filename, delimiter=";", header=0)
    datapoints_df = pd.read_csv(points_filename, delimiter=";", header=0)
    conditions_df = pd.read_csv("data/Task2_conditions_updated.csv", delimiter=",", header=0)

    conditions_df["Parameters_deafult"] = conditions_df["Parameters"].apply(lambda x: x.split("] ")[0])
    conditions_df["Parameters"] = conditions_df["Parameters"].apply(lambda x: x.replace("] ", " "))

    # print(conditions_df.columns)
    range_of_conditions = dataset_df["Number of conditions"].unique()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(output_dir + "/" + model_name):
        os.mkdir(output_dir + "/" + model_name)

    idx = 0
    for no_of_conditions in range_of_conditions:
        python_code_path = output_dir + "/" + model_name + "/" + str(no_of_conditions) + "/python_code"
        datapoints_path = output_dir + "/" + model_name + "/" + str(no_of_conditions) + "/datapoints"

        if not os.path.exists(output_dir + "/" + model_name + "/" + str(no_of_conditions)):
            os.mkdir(output_dir + "/" + model_name + "/" + str(no_of_conditions))

        if not os.path.exists(python_code_path):
            os.mkdir(python_code_path)

        if not os.path.exists(datapoints_path):
            os.mkdir(datapoints_path)

            # samples_per_param = dataset_df["Number of parameters"].value_counts()[no_of_parameters]

        # param_outputs_df = outputs_df[idx:idx+samples_per_param]
        param_outputs_df = outputs_df[outputs_df["Number of conditions"] == no_of_conditions]
        # idx += samples_per_param
        for idx2 in param_outputs_df[param_outputs_df["Syntax eval"] == "Correct syntax"].index:
            input_datapoints = {"Parameters": [], "Datapoints": [], "Outlier": []}

            # Get a Python code generated by a model
            python_output = outputs_df["Model output"][idx2] + "\n"
            # all_comments = re.findall(r"(#.*\n)", python_output)
            # rules = re.findall(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*->\s*[a-zA-Z_][a-zA-Z0-9_]*:\n(?: {4}.+\n?)+", python_output)

            # if len(all_comments) != len(rules):
            #     python_code = "\n".join(rules)
            # else:
            #     python_code = ""
            #     for comment, rule in zip(all_comments, rules):
            #         python_code += comment + rule + "\n"
            main_rule_no = dataset_df["Number of conditions"][idx2]+1
            main_rule = f"r{main_rule_no}"
            # datapoints for each parameter = {no-outlier, no-outlier, outlier}
            # TODO remove from [idx*50] 50
            prompt_parameters = ast.literal_eval(dataset_df["Parameters"][idx2])
            # print(f"Prompt parameters:\n{prompt_parameters}")

            # Saving GT and results of prediction
            # First iterate over 3 validation set of points
            for j in range(3):
                # Define a template for evaluating and printing results of each validation set
                # print(r(n+1)(arg1, arg2, ..., argM))
                eval_line = f"\nprint({main_rule}([args]))"
                eval_arguments = ""
                # Iterate over each condition's parameters in a prompt
                for condition_params in prompt_parameters:
                    # print(f"Condition parameters:\n{condition_params}")
                    condition_params_str = ", ".join(condition_params)
                    # Find the index of a condition which is actually preprocessed - range of indices == <0; 10>
                    # cond_idx = conditions_df[conditions_df["Parameters_deafult"].str.contains(cond_params)].index[0]
                    condition_idx = conditions_df[conditions_df["Parameters"].str.contains(condition_params_str)].index[0]
                    # We select a validation dataset and outlier as:
                    # condition_index * 3 (because each condition has 3 validation steps) + step of validation (from 1 to 3)
                    cond_datapoints = ast.literal_eval(datapoints_df["Values"][condition_idx * 3 + j])
                    cond_outlier = datapoints_df["Outlier"][condition_idx * 3 + j]

                    # if len(cond_datapoints) == 4:
                    # print(f"Before: {cond_datapoints}")

                    if condition_params_str != conditions_df["Parameters_deafult"][condition_idx]:
                        # print(f"--------------------------- Example {idx2} ----------------------------")
                        # print(f"Current params: {condition_params_str}")
                        # print(conditions_df["Parameters_deafult"][condition_idx])
                        # print("Cond datapoint before:", cond_datapoints)
                        # Move the first parameter
                        param_to_move = cond_datapoints.pop(0)
                        cond_datapoints.append(param_to_move)
                        # print("Cond datapoint after:", cond_datapoints)
                        # print(f"------------------------------------------------------------------------")

                    # cond_datapoints = ", ".join(cond_datapoints)

                    # print(f"After: {cond_datapoints}")

                    input_datapoints["Parameters"].append(condition_params)
                    input_datapoints["Datapoints"].append(cond_datapoints)
                    input_datapoints["Outlier"].append(cond_outlier)
           
                    # Add each datapoint to the arguments of final function call
                    # print(cond_datapoints)
                    for datapoint in cond_datapoints:
                        eval_arguments += str(datapoint) + ", "

                python_output += eval_line.replace("[args]", eval_arguments[:-2]) + "\n"
            
            input_datapoints_df = pd.DataFrame(input_datapoints)
            input_datapoints_df.to_csv(datapoints_path  + "/" + str(idx2) + "_datapoints.csv", sep=";", columns=list(input_datapoints_df.keys()), index=False)

            f = open(python_code_path + "/" + str(idx2) + "_output_python.py", "w")
            f.write(python_output)
            f.close()


def evaluate_outputs(task, model_name, Lean_dir_all, report_dict):
    # eval_results = {"Parameters": []}

    # print(glob.glob("Lean_outputs//*"))
    # print(Lean_dir_all)
    # print(glob.glob(f"{Lean_dir_all}/*"))
    params = [int(param.split('/')[-1]) for param in glob.glob(f"{Lean_dir_all}/{model_name}/*")]
    # print(params)
    for param in tqdm(sorted(params), desc="Iteration over params"):
        python_codes = sorted(glob.glob(f"{Lean_dir_all}/{model_name}/{str(param)}/python_code/*.py"))
        datapoints = sorted(glob.glob(f"{Lean_dir_all}/{model_name}/{str(param)}/datapoints/*.csv"))

        for python_code, eval_datapoints in tqdm(zip(python_codes, datapoints), desc="Iteration over examples"):
            prompt_no = int(eval_datapoints.split("/")[-1].split("_")[0])
                             
            # Retrieve answers from a python eval code
            python_output_obj = subprocess.run(['python', f'{python_code}'], capture_output=True, text=True)
            # Retrieve standard output
            python_output = python_output_obj.stdout
            # Retrieve standard error
            python_err = python_output_obj.stderr

            # print(lean_output)
            wrong_rules = False
            answers = []

            error_answer = ''

            # Check if execution of a code triggered an error
            if python_output == '':
                # This idicates execution error - means that rules are syntactically correct, but takes wrong arguments while evaluating i.e. 1 parameter instead of 2
                # So we must to check standard error output
                error_answer = python_err
                # print(error_answer)
                print(f"{python_code} has wrong rules (empty stdout)")
                # continue

            # Retrieve all answers for each validation set of points
            # return not(r1(arg1, arg2) and r2(arg3, arg4)), r1(arg1, arg2), r2(arg3, arg4)
            # print(r3(arg1, arg2, arg3, arg4))
            # Result: False, True, True
            if error_answer == '':
                for line_of_answers in python_output.split('\n')[:-1]:
                    eval_answers = line_of_answers[1:-1].split(', ')
                    # print(eval_answers)
                    if len(eval_answers) == param + 1:
                        answers.append(", ".join(eval_answers))

            # if wrong_rules == True:
            #     print(f"{python_code} has wrong rules")
            #     # print(lean_code)
            #     # Go to the next code example
            #     continue

            # if len(answers) != 3:
            #     print(f"{python_code} has wrong number of answers")
            #     continue
            
            is_correct = True
            # print(answers)
            # Retrieve ground truth values about outliers
            eval_datapoints_df = pd.read_csv(eval_datapoints, delimiter=";", header=0)
            df_columns = eval_datapoints_df.columns
            # datapoints for each parameter = [outlier_low, no-outlier_low, no-outlier_high, outlier_high]
            # answers_idx = 0
            if task == "Task1":
                # TODO watch this part of code, looking for potential errors in logic
                if error_answer != '':
                    print(f"Error answer:{error_answer}")
                    report_dict["Outlier detection"][prompt_no] = error_answer
                else:
                    gt_answers = []
                    for idx1 in [0, 3]:
                        for idx2 in [1, 2]:
                            for param_outlier1 in df_columns[1::2]:
                                gt_line_of_answers = ""
                                gt_is_outlier = True
                                for param_outlier2 in df_columns[1::2]:
                                    if param_outlier1 == param_outlier2:
                                        point_is_outlier = not bool(eval_datapoints_df[param_outlier2][idx1])
                                    else:
                                        point_is_outlier = not bool(eval_datapoints_df[param_outlier2][idx2])

                                    gt_line_of_answers += f"{point_is_outlier}, "
                                    # print(f"point_is_outlier: {point_is_outlier}")
                                    # print(f"gt_line_of_answers: {gt_line_of_answers}")
                                    gt_is_outlier = gt_is_outlier and point_is_outlier

                                gt_line_of_answers = f"{not gt_is_outlier}, {gt_line_of_answers}"
                                gt_answers.append(gt_line_of_answers[:-2])                     
                    
                    report_dict["Outlier"][prompt_no] = gt_answers
                    report_dict["Outlier detection"][prompt_no] = answers

            elif task == "Task2":
                # print(error_answer)
                if error_answer != '':
                    print(f"Error answer:{error_answer}")
                    report_dict["Outlier detection"][prompt_no] = error_answer
                else:
                    print(eval_datapoints)
                    is_outlier = False
                    gt_answers = []
                    # Iterate over each validation call
                    for val_step in range(3):                
                        gt_line_of_answers = ""
                        gt_is_outlier = True
                        # Iterate over set of validation points for each condition/function
                        for idx in range(param):
                            # Get a GT answer for rule number (idx) with negation, because rules returns True if point is not an outlier
                            rule_call_gt = not bool(eval_datapoints_df["Outlier"][val_step*param + idx])
                            # Agregate for final answer
                            gt_is_outlier = gt_is_outlier and rule_call_gt
                            # Add to the gt answers array for final report
                            gt_line_of_answers += f"{rule_call_gt}, "

                        gt_is_outlier = not gt_is_outlier
                        gt_line_of_answers = f"{gt_is_outlier}, {gt_line_of_answers}"
                        # To remove ", " at the end of the string
                        gt_answers.append(gt_line_of_answers[:-2])

                    # print(answers)
                    report_dict["Outlier"][prompt_no] = gt_answers
                    report_dict["Outlier detection"][prompt_no] = answers



def generate_full_validation_report(is_revised, task, model_name, model_outputs, output_dir, premises_dataset, points_dataset, report_filename):
    model_outputs_df = pd.read_csv(model_outputs, keep_default_na=False, delimiter=";", header=0)

    if not is_revised:
        report_dict = {"Prompt": [], "Premises": [], "No. of parameters": [], "Parameters": [], "Model": [], "Model output": [], "Syntax eval": [], "Outlier": [], "Outlier detection": []}

        print(f"Evaluation of model: {model_name}")
        #Load a csv file with outputs from LLMs
        print(f"Syntax evaluation")
        evaluate_outputs_syntax(task, is_revised, model_name, model_outputs_df, report_dict, premises_dataset, points_dataset, system_prompt="")
    
    else:
        report_dict = model_outputs_df.to_dict()
        print(f"Evaluation of model: {model_name}")
        #Load a csv file with outputs from LLMs
        evaluate_outputs_syntax(task, is_revised, model_name, model_outputs_df, report_dict, premises_dataset, points_dataset, system_prompt="")       
    
    print(f"Saving outputs")

    if task == "Task1":
        save_outputs_task1(model_name, report_dict, dataset_filename="data/dataset_premises.csv",
             points_filename="data/points_dataset.csv", output_dir=output_dir)
    else:
        save_outputs_task2(model_name, report_dict, dataset_filename="data/upgraded_task2_dataset_premises.csv",
                           points_filename="data/task2_points_dataset.csv", output_dir=output_dir)
    # # return
    # report_dict = pd.read_csv("results/New_val_Adapt_Llama-70_results.csv", delimiter=";", header=0)
    print(f"Rules evaluation\n")
    evaluate_outputs(task=task, model_name=model_name, Lean_dir_all=output_dir, report_dict=report_dict)

    report_dict_df = pd.DataFrame(report_dict)
    report_dict_df.to_csv(report_filename, sep=";", columns=list(report_dict_df.keys()), index=False)


def calculate_metrics(report_filenames_expr, results_filename):
    report_filenames = glob.glob(report_filenames_expr)

    metrics_results = {"Model": [], "Formalization": [], "Correctness": [], "Overall": []}

    for report_filename in report_filenames:
        model_name = report_filename.split("/")[-1].split("_")[3]
        # if model_name == "critic":
        #     model_name = report_filename.split("/")[-1].split("_")[-4]

        report_df = pd.read_csv(report_filename, delimiter=";", header=0)

        metrics_results["Model"].append(model_name)

        # Calculate Formalization Accuracy (FA)
        correct_syntax_outputs_df = report_df[report_df["Syntax eval"] == "Correct syntax"]
        no_correct_syntax = len(correct_syntax_outputs_df.index)
        FA = round(no_correct_syntax / len(report_df.index), 2)

        metrics_results["Formalization"].append(FA)

        # Calculate Correctness Accuracy (CA)
        # gt_answers = report_df["Outlier"]
        # TODO remove cases when both columns have values "Error"
        correct_rules_df = report_df[report_df["Outlier"] == report_df["Outlier detection"]]
        no_correct_rules = len(correct_rules_df.index)
        CA = round(no_correct_rules / no_correct_syntax, 2)   

        metrics_results["Correctness"].append(CA)

        # Calculate Overall Accuracy (OA)
        OA = round(no_correct_rules / len(report_df.index), 2)

        metrics_results["Overall"].append(OA)

    metrics_results_df = pd.DataFrame(metrics_results)
    metrics_results_df = metrics_results_df.sort_values(by=["Overall"], ascending=False)
    print(metrics_results_df["Overall"])
    metrics_results_df.to_csv(results_filename, sep=";", columns=list(metrics_results_df.keys()), index=False)


def update_outputs(model_name, model_report, original_model_outputs, corrected_model_outputs, results_filename):
    model_report_df = pd.read_csv(model_report, delimiter=";", header=0)
    original_model_outputs_df = pd.read_csv(original_model_outputs, delimiter=";", header=0)
    corrected_model_outputs_df = pd.read_csv(corrected_model_outputs, delimiter=";", header=0)

    indices_to_replace = model_report_df[model_report_df["Syntax eval"] != "Correct syntax"].index

    print(indices_to_replace)

    idx2 = 0
    for idx in indices_to_replace:
        original_model_outputs_df[model_name][idx] = corrected_model_outputs_df[f"{model_name} new output"][idx2]


    original_model_outputs_df.to_csv(results_filename, sep=";", columns=list(original_model_outputs_df.keys()), index=False)


# update_outputs("Llama-70", "results/New_eval/New_val_Adapt_Llama-70_results.csv", "outputs/Adapt_anonym_Llama-70_outputs.csv", "outputs/critic_syntax/task1_Llama-70_new_val_critic_outputs.csv", "outputs/Adapt_anonym_Llama-70_outputs_syntax.csv")
# update_outputs("Llama-8", "results/New_eval/New_val_Adapt_Llama-8_results.csv", "outputs/Adapt_anonym_Llama-8_outputs.csv", "outputs/critic_syntax/task1_Llama-8_new_val_critic_outputs.csv", "outputs/Adapt_anonym_Llama-8_outputs_syntax.csv")
# update_outputs("Mistral", "results/New_eval/New_val_Adapt_Mistral_results.csv", "outputs/Adapt_anonym_Mistral_outputs.csv", "outputs/critic_syntax/task1_Mistral_new_val_critic_outputs.csv", "outputs/Adapt_anonym_Mistral_outputs_syntax.csv")
# update_outputs("Mixtral", "results/New_eval/New_val_Adapt_Mixtral_results.csv", "outputs/Adapt_anonym_Mixtral_outputs.csv", "outputs/critic_syntax/task1_Mixtral_new_val_critic_outputs.csv", "outputs/Adapt_anonym_Mixtral_outputs_syntax.csv")
# update_outputs("Qwen", "results/New_eval/New_val_Adapt_Qwen_results.csv", "outputs/Adapt_anonym_Qwen_outputs.csv", "outputs/critic_syntax/task1_Qwen_new_val_critic_outputs.csv", "outputs/Adapt_anonym_Qwen_outputs_syntax.csv")
# update_outputs("Gemma", "results/New_eval/New_val_Adapt_Gemma_results.csv", "outputs/Adapt_anonym_Gemma_outputs.csv", "outputs/critic_syntax/task1_Gemma_new_val_critic_outputs.csv", "outputs/Adapt_anonym_Gemma_outputs_syntax.csv")

# generate_full_validation_report(False, "Task1", "Llama-70", "outputs/Adapt_anonym_Llama-70_python_outputs.csv", "New_val_Adapt_Task1_python_outputs_first", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-70_first_python_results.csv")
# generate_full_validation_report(False, "Task1", "Llama-8", "outputs/Adapt_anonym_Llama-8_python_outputs.csv", "New_val_Adapt_Task1_python_outputs_first", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-8_first_python_results.csv")
# generate_full_validation_report(False, "Task1", "Mistral", "outputs/Adapt_anonym_Mistral_python_outputs.csv", "New_val_Adapt_Task1_python_outputs_first", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Mistral_first_python_results.csv")
# generate_full_validation_report(False, "Task1", "Mixtral", "outputs/Adapt_anonym_Mixtral_python_outputs.csv", "New_val_Adapt_Task1_python_outputs_first", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Mixtral_first_python_results.csv")
# generate_full_validation_report(False, "Task1", "Qwen", "outputs/Adapt_anonym_Qwen_python_outputs.csv", "New_val_Adapt_Task1_python_outputs_first", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Qwen_first_python_results.csv")
# generate_full_validation_report(False, "Task1", "Gemma", "outputs/Adapt_anonym_Gemma_python_outputs.csv", "New_val_Adapt_Task1_python_outputs_first", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Gemma_first_python_results.csv")

# calculate_metrics("results/New_eval/*_first_python_results.csv", "metrics/new_Adapt_task1_first_python_metrics.csv")

# generate_full_validation_report(False, "Task1", "Llama-70", "outputs/Adapt_anonym_Llama-70_outputs_syntax.csv", "New_val_Adapt_Task1_outputs_critic_syntax", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-70_critic_syntax_results.csv")
# generate_full_validation_report(False, "Task1", "Llama-8", "outputs/Adapt_anonym_Llama-8_outputs_syntax.csv", "New_val_Adapt_Task1_outputs_critic_syntax", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-8_critic_syntax_results.csv")
# generate_full_validation_report(False, "Task1", "Mistral", "outputs/Adapt_anonym_Mistral_outputs_syntax.csv", "New_val_Adapt_Task1_outputs_critic_syntax", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Mistral_critic_syntax_results.csv")
# generate_full_validation_report(False, "Task1", "Mixtral", "outputs/Adapt_anonym_Mixtral_outputs_syntax.csv", "New_val_Adapt_Task1_outputs_critic_syntax", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Mixtral_critic_syntax_results.csv")
# generate_full_validation_report(False, "Task1", "Qwen", "outputs/Adapt_anonym_Qwen_outputs_syntax.csv", "New_val_Adapt_Task1_outputs_critic_syntax", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Qwen_critic_syntax_results.csv")
# generate_full_validation_report(False, "Task1", "Gemma", "outputs/Adapt_anonym_Gemma_outputs_syntax.csv", "New_val_Adapt_Task1_outputs_critic_syntax", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Gemma_critic_syntax_results.csv")

# calculate_metrics("results/New_eval/*_critic_syntax_results.csv", "metrics/new_Adapt_task1_critic_syntax_metrics.csv")

# generate_full_validation_report(True, "Task1", "Llama-70", "outputs/task1_Llama-70_new_val_critic_rules_outputs.csv", "New_val_Adapt_Task1_outputs_critic_rules", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-70_critic_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Llama-8", "outputs/task1_Llama-8_new_val_critic_rules_python_outputs.csv", "New_val_Adapt_Task1_outputs_critic_python_rules", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-8_critic_python_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Mistral", "outputs/task1_Mistral_new_val_critic_rules_python_outputs.csv", "New_val_Adapt_Task1_outputs_critic_python_rules", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Mistral_critic_python_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Mixtral", "outputs/task1_Mixtral_new_val_critic_rules_outputs.csv", "New_val_Adapt_Task1_outputs_critic_rules", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Mixtral_critic_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Qwen", "outputs/task1_Qwen_new_val_critic_rules_python_outputs.csv", "New_val_Adapt_Task1_outputs_critic_python_rules", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Qwen_critic_python_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Gemma", "outputs/task1_Gemma_new_val_critic_rules_python_outputs.csv", "New_val_Adapt_Task1_outputs_critic_python_rules", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Gemma_critic_python_rules_results.csv")

# generate_full_validation_report(True, "Task1", "Llama-8", "outputs/task1_Llama-8_new_val_critic_rules_python_outputs_incontext_upgrade.csv", "New_val_Adapt_Task1_outputs_critic_python_rules_incontext_upgrade", "data/dataset_premises.csv", "data/points_dataset.csv", "results/New_eval/New_val_Adapt_Llama-8_critic_python_rules_incontext_upgrade_results.csv")
# calculate_metrics("results/New_eval/*_critic_python_rules_incontext_upgrade_results.csv", "metrics/new_Adapt_task1_critic_python_rules_metrics.csv")

# generate_full_validation_report(False, "Task2", "Llama-8", "new_outputs/Python_task2/Adapt_anonym_Llama-8_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/first_iter/task2/New_val_Adapt_Llama-8_task2_python_results_no_final_rule.csv")
# generate_full_validation_report(False, "Task2", "Llama-70", "new_outputs/Python_task2/Adapt_anonym_Llama-70_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/first_iter/task2/New_val_Adapt_Llama-70_task2_python_results_no_final_rule.csv")
# generate_full_validation_report(False, "Task2", "Mistral", "new_outputs/Python_task2/Adapt_anonym_Mistral_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/first_iter/task2/New_val_Adapt_Mistral_task2_python_results_no_final_rule.csv")
# generate_full_validation_report(False, "Task2", "Mixtral", "new_outputs/Python_task2/Adapt_anonym_Mixtral_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/first_iter/task2/New_val_Adapt_Mixtral_task2_python_results_no_final_rule.csv")
# generate_full_validation_report(False, "Task2", "Qwen", "new_outputs/Python_task2/Adapt_anonym_Qwen_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/first_iter/task2/New_val_Adapt_Qwen_task2_python_results_no_final_rule.csv")
# generate_full_validation_report(False, "Task2", "Gemma", "new_outputs/Python_task2/Adapt_anonym_Gemma_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/first_iter/task2/New_val_Adapt_Gemma_task2_python_results_no_final_rule.csv")

# calculate_metrics("results/first_iter/task2/New_val_Adapt_*_task2_python_results_no_final_rule.csv", "metrics/Python/New_val_Adapt_task2_python_rules_no_final_rule_metrics.csv")

# generate_full_validation_report(True, "Task2", "Llama-70", "new_outputs/Python_task2/Adapt_anonym_Llama-70_critic_runtime_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_runtime_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/runtime/task2/New_val_Adapt_Llama-70_task2_critic_runtime_python_results_no_final_rule.csv")
# generate_full_validation_report(True, "Task2", "Llama-8", "new_outputs/Python_task2/Adapt_anonym_Llama-8_critic_runtime_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_runtime_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/runtime/task2/New_val_Adapt_Llama-8_task2_critic_runtime_python_results_no_final_rule.csv")
# generate_full_validation_report(True, "Task2", "Mixtral", "new_outputs/Python_task2/Adapt_anonym_Mixtral_critic_runtime_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_runtime_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/runtime/task2/New_val_Adapt_Mixtral_task2_critic_runtime_python_results_no_final_rule.csv")
# generate_full_validation_report(True, "Task2", "Mistral", "new_outputs/Python_task2/Adapt_anonym_Mistral_critic_runtime_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_runtime_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/runtime/task2/New_val_Adapt_Mistral_task2_critic_runtime_python_results_no_final_rule.csv")
# generate_full_validation_report(True, "Task2", "Gemma", "new_outputs/Python_task2/Adapt_anonym_Gemma_critic_runtime_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_runtime_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/runtime/task2/New_val_Adapt_Gemma_task2_critic_runtime_python_results_no_final_rule.csv")
# generate_full_validation_report(True, "Task2", "Qwen", "new_outputs/Python_task2/Adapt_anonym_Qwen_critic_runtime_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_runtime_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/runtime/task2/New_val_Adapt_Qwen_task2_critic_runtime_python_results_no_final_rule.csv")


# calculate_metrics("results/runtime/task2/New_val_Adapt_*_task2_critic_runtime_python_results_no_final_rule.csv", "metrics/Python/New_val_Adapt_task2_critic_runtime_python_rules_no_final_rule_metrics.csv")

generate_full_validation_report(True, "Task2", "Llama-70", "new_outputs/Python_task2/Adapt_anonym_Llama-70_critic_rules_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_rules_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/critic_rules/task2/New_val_Adapt_Llama-70_task2_critic_rules_python_results_no_final_rule.csv")
generate_full_validation_report(True, "Task2", "Llama-8", "new_outputs/Python_task2/Adapt_anonym_Llama-8_critic_rules_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_rules_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/critic_rules/task2/New_val_Adapt_Llama-8_task2_critic_rules_python_results_no_final_rule.csv")
generate_full_validation_report(True, "Task2", "Mixtral", "new_outputs/Python_task2/Adapt_anonym_Mixtral_critic_rules_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_rules_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/critic_rules/task2/New_val_Adapt_Mixtral_task2_critic_rules_python_results_no_final_rule.csv")
generate_full_validation_report(True, "Task2", "Mistral", "new_outputs/Python_task2/Adapt_anonym_Mistral_critic_rules_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_rules_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/critic_rules/task2/New_val_Adapt_Mistral_task2_critic_rules_python_results_no_final_rule.csv")
generate_full_validation_report(True, "Task2", "Gemma", "new_outputs/Python_task2/Adapt_anonym_Gemma_critic_rules_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_rules_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/critic_rules/task2/New_val_Adapt_Gemma_task2_critic_rules_python_results_no_final_rule.csv")
generate_full_validation_report(True, "Task2", "Qwen", "new_outputs/Python_task2/Adapt_anonym_Qwen_critic_rules_task2_python_outputs_no_final_rule.csv", "New_val_Adapt_Task2_outputs_python_critic_rules_no_final_rule", "data/upgraded_task2_dataset_premises.csv", "data/task2_points_dataset.csv", "results/critic_rules/task2/New_val_Adapt_Qwen_task2_critic_rules_python_results_no_final_rule.csv")


calculate_metrics("results/critic_rules/task2/New_val_Adapt_*_task2_critic_rules_python_results_no_final_rule.csv", "metrics/Python/New_val_Adapt_task2_critic_rules_python_rules_no_final_rule_metrics.csv")

# generate_full_validation_report(False, "Task1", "Llama-70", "outputs/Python_task1/Adapt_anonym_Llama-70_python_outputs.csv", "New_val_Adapt_Task1_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/first_iter/task1/New_val_Adapt_Llama-70_results.csv")
# generate_full_validation_report(False, "Task1", "Llama-8", "outputs/Python_task1/Adapt_anonym_Llama-8_python_outputs.csv", "New_val_Adapt_Task1_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/first_iter/task1/New_val_Adapt_Llama-8_results.csv")
# generate_full_validation_report(False, "Task1", "Mistral", "outputs/Python_task1/Adapt_anonym_Mistral_python_outputs.csv", "New_val_Adapt_Task1_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/first_iter/task1/New_val_Adapt_Mistral_results.csv")
# generate_full_validation_report(False, "Task1", "Mixtral", "outputs/Python_task1/Adapt_anonym_Mixtral_python_outputs.csv", "New_val_Adapt_Task1_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/first_iter/task1/New_val_Adapt_Mixtral_results.csv")
# generate_full_validation_report(False, "Task1", "Qwen", "outputs/Python_task1/Adapt_anonym_Qwen_python_outputs.csv", "New_val_Adapt_Task1_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/first_iter/task1/New_val_Adapt_Qwen_results.csv")
# generate_full_validation_report(False, "Task1", "Gemma", "outputs/Python_task1/Adapt_anonym_Gemma_python_outputs.csv", "New_val_Adapt_Task1_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/first_iter/task1/New_val_Adapt_Gemma_results.csv")

# calculate_metrics("results/first_iter/task1/New_val_Adapt_*_results.csv", "metrics/Python/New_val_Adapt_task1_first_python_rules_metrics.csv")

# generate_full_validation_report(True, "Task1", "Llama-70", "outputs/Python_task1/Adapt_anonym_Llama-70_critic_syntax_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_syntax_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/syntax/task1/New_val_Adapt_Llama-70_critic_syntax_results.csv")
# generate_full_validation_report(True, "Task1", "Llama-8", "outputs/Python_task1/Adapt_anonym_Llama-8_critic_syntax_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_syntax_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/syntax/task1/New_val_Adapt_Llama-8_critic_syntax_results.csv")
# generate_full_validation_report(True, "Task1", "Mistral", "outputs/Python_task1/Adapt_anonym_Mistral_critic_syntax_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_syntax_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/syntax/task1/New_val_Adapt_Mistral_critic_syntax_results.csv")
# generate_full_validation_report(True, "Task1", "Mixtral", "outputs/Python_task1/Adapt_anonym_Mixtral_critic_syntax_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_syntax_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/syntax/task1/New_val_Adapt_Mixtral_critic_syntax_results.csv")
# generate_full_validation_report(True, "Task1", "Qwen", "outputs/Python_task1/Adapt_anonym_Qwen_critic_syntax_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_syntax_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/syntax/task1/New_val_Adapt_Qwen_critic_syntax_results.csv")
# generate_full_validation_report(True, "Task1", "Gemma", "outputs/Python_task1/Adapt_anonym_Gemma_critic_syntax_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_syntax_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/syntax/task1/New_val_Adapt_Gemma_critic_syntax_results.csv")

# calculate_metrics("results/syntax/task1/New_val_Adapt_*_critic_syntax_results.csv", "metrics/Python/New_val_Adapt_task1_syntax_python_rules_metrics.csv")

# generate_full_validation_report(True, "Task1", "Llama-70", "outputs/Python_task1/Adapt_anonym_Llama-70_critic_runtime_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_runtime_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Llama-70_critic_runtime_results.csv")
# generate_full_validation_report(True, "Task1", "Llama-8", "outputs/Python_task1/Adapt_anonym_Llama-8_critic_runtime_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_runtime_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Llama-8_critic_runtime_results.csv")
# generate_full_validation_report(True, "Task1", "Mistral", "outputs/Python_task1/Adapt_anonym_Mistral_critic_runtime_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_runtime_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Mistral_critic_runtime_results.csv")
# generate_full_validation_report(True, "Task1", "Mixtral", "outputs/Python_task1/Adapt_anonym_Mixtral_critic_runtime_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_runtime_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Mixtral_critic_runtime_results.csv")
# generate_full_validation_report(True, "Task1", "Qwen", "outputs/Python_task1/Adapt_anonym_Qwen_critic_runtime_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_runtime_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Qwen_critic_runtime_results.csv")
# generate_full_validation_report(True, "Task1", "Gemma", "outputs/Python_task1/Adapt_anonym_Gemma_critic_runtime_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_runtime_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Gemma_critic_runtime_results.csv")

# calculate_metrics("results/runtime/task1/New_val_Adapt_*_critic_runtime_results.csv", "metrics/Python/New_val_Adapt_task1_runtime_python_rules_metrics.csv")

# generate_full_validation_report(True, "Task1", "Llama-8", "outputs/Python_task1/Adapt_anonym_Llama-8_critic_rules_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_rules_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Llama-8_critic_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Mistral", "outputs/Python_task1/Adapt_anonym_Mistral_critic_rules_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_rules_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Mistral_critic_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Mixtral", "outputs/Python_task1/Adapt_anonym_Mixtral_critic_rules_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_rules_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Mixtral_critic_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Qwen", "outputs/Python_task1/Adapt_anonym_Qwen_critic_rules_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_rules_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Qwen_critic_rules_results.csv")
# generate_full_validation_report(True, "Task1", "Gemma", "outputs/Python_task1/Adapt_anonym_Gemma_critic_rules_task1_python_outputs_no_final_rule.csv", "New_val_Adapt_Task1_critic_rules_outputs", "data/dataset_premises.csv", "data/points_dataset.csv", "results/runtime/task1/New_val_Adapt_Gemma_critic_rules_results.csv")

# calculate_metrics("results/runtime/task1/New_val_Adapt_*_critic_rules_results.csv", "metrics/Python/New_val_Adapt_task1_rules_python_rules_metrics.csv")