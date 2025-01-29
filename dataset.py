import re
import ast
import random
import itertools
import pandas as pd


def generate_dataset_task1_llm():
    dataset_df = pd.read_csv("data/dataset_premises.csv", delimiter=";", header=0)
    datapoints_df = pd.read_csv("data/points_dataset.csv", delimiter=";", header=0)
    # Retrieve parameters, and values for parameters

    new_dataset = {"Prompt": [], "Answer": []}

    for idx in dataset_df.index:
        prompt = dataset_df["Prompt"][idx]
        used_params = ""
        param_values = ""
        is_outlier = False

        for param_name in dataset_df.columns[2:]:
            if dataset_df[param_name][idx] == 1:
                used_params += param_name + ", "
                param_idx = random.choice(datapoints_df[datapoints_df["Parameter"] == param_name].index)
                # print(param_idx)

                param_unit = ", " if datapoints_df["Unit"][param_idx] == "-" else datapoints_df["Unit"][param_idx] + ", "
                if param_unit != ", ":
                    param_value = round(datapoints_df["Value"][param_idx], 1)
                else:
                    param_value = int(datapoints_df["Value"][param_idx])

                param_values += param_name + ": " + str(param_value) + " " + param_unit

                if datapoints_df["Outlier"][param_idx] == 1:
                    is_outlier = True

        
        new_prompt = """Evaluate the following premises:
[[Prompt]]
Used parameters: [[Params]].
Data points for used parameters: [[Data points]].
To identify the day as abnormal, it is enough that even one or more conditions are violated.
Remember to return at the end True if the day is abnormal or False otherwise.
"""

        new_prompt = new_prompt.replace("[[Prompt]]", prompt).replace("[[Params]]", used_params[:-2]).replace("[[Data points]]", param_values[:-2])

        new_dataset["Prompt"].append(new_prompt)
        new_dataset["Answer"].append(is_outlier)
                
        new_dataset_df = pd.DataFrame(new_dataset)
        new_dataset_df.to_csv("data/llm_only_dataset_premises.csv", sep=";", columns=list(new_dataset.keys()), index=False)


def old_generate_dataset_task2_llm():
    dataset_df = pd.read_csv("data/task2_dataset_premises.csv", delimiter=";", header=0)
    datapoints_df = pd.read_csv("data/task2_points_dataset.csv", delimiter=";", header=0)
    parameters_df = pd.read_csv("data/parameters.csv", delimiter=";", header=0)
    # Retrieve parameters, and values for parameters

    new_dataset = {"Prompt": [], "Answer": []}

    for idx in dataset_df.index:
        prompt = dataset_df["Prompt"][idx]
        used_params = ""
        param_values = ""
        is_outlier = False

        data_points = ast.literal_eval(datapoints_df["Values"][idx])

        prompt_parameters_lists = ast.literal_eval(dataset_df["Parameters"][idx])
        prompt_parameters = [param_name for parameters in prompt_parameters_lists for param_name in parameters]
        for param_name, param_value in zip(prompt_parameters, data_points):      
            used_params += param_name + ", "
            param_idx = parameters_df[parameters_df["Parameters"] == param_name].index[0]

            param_unit = ", " if parameters_df["Unit"][param_idx] == "-" else parameters_df["Unit"][param_idx] + ", "

            param_values += param_name + ": " + str(param_value) + " " + param_unit

        if datapoints_df["Outlier"][idx] == 1:
            is_outlier = True

        
        new_prompt = """Evaluate the following premises:
[[Prompt]]
Used parameters: [[Params]].
Data points for used parameters: [[Data points]].
To identify the day as abnormal, it is enough that even one or more conditions are violated.
Remember to return at the end True if the day is abnormal or False otherwise.
"""

        new_prompt = new_prompt.replace("[[Prompt]]", prompt).replace("[[Params]]", used_params[:-2]).replace("[[Data points]]", param_values[:-2])

        new_dataset["Prompt"].append(new_prompt)
        new_dataset["Answer"].append(is_outlier)
                
    new_dataset_df = pd.DataFrame(new_dataset)
    new_dataset_df.to_csv("data/task2_llm_only_dataset_premises.csv", sep=";", columns=list(new_dataset.keys()), index=False)


def generate_dataset():
    statements_df = pd.read_csv("data/statements.csv", delimiter=",", header=0,
                names=["Parameter", "Original statement", "Version 1", "Version 2", "Version 3", "Version 4", "Version 5"])
    
    dataset = {"Prompt": [],
               "Number of parameters": []}

    for param in statements_df["Parameter"]:
        dataset[param] = []

    # Get the total number of statements
    m_statements = len(statements_df.index)
    m_statements_array = [i for i in range(m_statements)]

    # Maximum range of number of parameters in a prompt
    m = 12

    # Number of prompts per number of parameters
    p = 50
    for i in range(2, m+1):
        # Generate all combinations of i parameters from 17
        m_combinations = list(itertools.combinations(m_statements_array, i))
        # print(len(m_combinations))
        # print(i)
        # print(p)
        # print("\n")
        # Select randomly 50 of these combinations
        selected_comb = random.sample(m_combinations, p)

        
        # Now select for each n parameters it's premise from 5 different syntaxes
        for j in range(p):
            set_of_premises = []
            parameters = selected_comb[j]
            
            for param in parameters:
                # Select a syntax's variant of a premise
                v = random.randint(1,5)
                version = "Version " + str(v)

                selected_premise = statements_df[version][param]

                set_of_premises.append(selected_premise)

            prompt = "Input text:\nTextual context: "
        
            for premise in set_of_premises:
                prompt += premise + "\n"

            # Add a prompt
            dataset["Prompt"].append(prompt)

            # Add a number of used parameters in a prompt (number of premises)
            dataset["Number of parameters"].append(len(parameters))

            # Additionally for each parameter indicate if this parameter has been selected for a prompt
            for i in statements_df.index:
                param = statements_df["Parameter"][i]
                if i in parameters:              
                    dataset[param].append(1)
                else:
                    dataset[param].append(0)



    dataset_prompts_df = pd.DataFrame(dataset)
    dataset_prompts_df.to_csv("data/dataset_premises.csv", sep=";", columns=list(dataset.keys()), index=False)


def old_generate_dataset_task2():
    statements_df = pd.read_csv("data/Task2_conditions_updated.csv", delimiter=";", header=0)
    
    params_df = pd.read_csv("data/parameters.csv", delimiter=";", header=0)
    
    dataset = {"Prompt": [], "Number of statements": [], "Parameters": []}

    # Get the total number of statements
    m_statements = len(statements_df.index)
    m_statements_array = [i for i in range(m_statements)]

    # Maximum number of compound sentences in a prompt
    m = 9

    # Number of prompts per number of parameters
    p = 50
    for i in range(2, m+1):
        # Generate all combinations of i parameters from 21
        m_combinations = list(itertools.combinations(m_statements_array, i))
        # Select randomly 50 of these combinations
        selected_comb = random.sample(m_combinations, p)

        # Now select for each n parameters it's premise from 5 different syntaxes
        for j in range(p):
            set_of_premises = []
            parameters = selected_comb[j]

            prompt_parameters = []
            
            for param in parameters:
                # Select used parameters in a statement
                used_params = statements_df["Parameters"][param].split(", ")
                # Add these parameters to the group of all parameters in a prompt
                prompt_parameters.append(used_params)
                # Select a syntax's variant of a premise
                v = random.randint(1,5)
                version = "Version " + str(v)

                selected_premise = statements_df[version][param]
                variables = re.findall("[XYZM]", selected_premise)

                fraction = random.random()
                # print(previous_range)
                if param == 20:
                    M_param = used_params[-1]
                    idx = params_df[params_df["Parameters"] == M_param].index[0]
                    param_min = params_df["Min"][idx]
                    param_max = params_df["Max"][idx]
                    M_param_value = round((param_max - param_min) * fraction + param_min, 1)
                    selected_premise = selected_premise.replace(variables[-1], str(M_param_value))

                    
                    # The sum of rest (3) parameters's values cannot be greather than value of a M parameter
                    # So the proportion have been defined in the following array
                    rest_params = [0.3, 0.3, 0.4]
                    
                    for rest_param, variable in zip(rest_params, variables[:-1]):
                        param_value = round(rest_param * M_param_value, 1)
                        selected_premise = selected_premise.replace(variable, str(param_value))
                else:
                    for used_param, variable in zip(used_params, variables):
                        idx = params_df[params_df["Parameters"] == used_param].index[0]
                        param_min = params_df["Min"][idx]
                        param_max = params_df["Max"][idx]
                        param_unit = params_df["Unit"][idx]
                        
                        param_value = round((param_max - param_min) * fraction + param_min, 1)

                        # If value must be integer, cast it to int
                        if param_unit == "-":
                            param_value = int(param_value)

                        selected_premise = selected_premise.replace(variable, str(param_value))


                set_of_premises.append(selected_premise)

            prompt = "Input text:\nTextual context: "
        
            for premise in set_of_premises:
                prompt += premise + "\n"

            prompt = prompt + "To identify the day as abnormal, it is enough that even one or more conditions are violated."
            # Add a prompt
            dataset["Prompt"].append(prompt)
            dataset["Number of statements"].append(len(parameters))
            dataset["Parameters"].append(prompt_parameters)

    dataset_prompts_df = pd.DataFrame(dataset)
    dataset_prompts_df.to_csv("data/task2_dataset_premises.csv", sep=";", columns=list(dataset.keys()), index=False)


def generate_dataset_task2():
    conditions_df = pd.read_csv("data/Task2_conditions_updated.csv", delimiter=",", header=0) 
    dataset = {"Prompt": [], "Number of conditions": [], "Parameters": []}

    # Get the total number of conditions
    m_conditions = len(conditions_df.index)
    m_conditions_array = [i for i in range(m_conditions)]

    # Maximum number of compound sentences in a prompt
    m = 9

    # Number of prompts per number of conditions in a single prompt
    p = 50
    for i in range(2, m+1):
        # Generate all combinations of i conditions from 11
        m_combinations = list(itertools.combinations(m_conditions_array, i))
        # Select randomly 50 of these combinations
        selected_comb = random.sample(m_combinations, p)

        # Now select for each n parameters it's premise from 5 different syntaxes
        for j in range(p):
            set_of_premises = []
            j_conditions = selected_comb[j]

            prompt_parameters = []
            
            for condition in j_conditions:
                # Select used parameters in a statement
                used_params = conditions_df["Parameters"][condition].split(", ")
                # Add these parameters to the group of all parameters in a prompt
                prompt_parameters.append(used_params)

                # Select a syntax's variant of a premise
                v = random.randint(1,5)
                version = "Version " + str(v)

                selected_premise = conditions_df[version][condition]

                set_of_premises.append(selected_premise)

            prompt = "Input text:\nTextual context: "
        
            for premise in set_of_premises:
                prompt += premise + "\n"

            prompt = prompt #+ "To identify the day as abnormal, it is enough that even one or more conditions are violated."
            # Add a prompt
            dataset["Prompt"].append(prompt)
            dataset["Number of conditions"].append(i)
            dataset["Parameters"].append(prompt_parameters)

    dataset_prompts_df = pd.DataFrame(dataset)
    dataset_prompts_df.to_csv("data/upgraded_task2_dataset_premises.csv", sep=";", columns=list(dataset.keys()), index=False)


def generate_dataset_task2_llm():
    conditions_df = pd.read_csv("data/Task2_conditions_updated.csv", delimiter=",", header=0)
    dataset_df = pd.read_csv("data/upgraded_task2_dataset_premises.csv", delimiter=";", header=0)
    datapoints_df = pd.read_csv("data/task2_points_dataset.csv", delimiter=";", header=0)
    parameters_df = pd.read_csv("data/parameters.csv", delimiter=";", header=0)
    # Retrieve parameters, and values for parameters

    new_dataset = {"Prompt": [], "Answer": []}

    for idx in dataset_df.index:
        prompt = dataset_df["Prompt"][idx]
        used_params = ""
        param_values = ""
        is_outlier = False

        # data_points = ast.literal_eval(datapoints_df["Values"][idx])
        prompt_parameters_lists = ast.literal_eval(dataset_df["Parameters"][idx])
        # prompt_parameters = [param_name for parameters in prompt_parameters_lists for param_name in parameters]
        for condition_params in prompt_parameters_lists:
            valid_idx = random.randint(0,2)
            condition_params_str = ", ".join(condition_params)
            condition_idx = conditions_df[conditions_df["Parameters"] == condition_params_str].index[0]
            condition_data_points = ast.literal_eval(datapoints_df["Values"][condition_idx * 3 + valid_idx])
            condition_outlier = int(datapoints_df["Outlier"][condition_idx * 3 + valid_idx])

            is_outlier = is_outlier and condition_outlier

            for param_name, param_value in zip(condition_params, condition_data_points):
                used_params += param_name + ", "
                param_idx = parameters_df[parameters_df["Parameters"] == param_name].index[0]

                param_unit = ", " if parameters_df["Unit"][param_idx] == "-" else parameters_df["Unit"][param_idx] + ", "

                param_values += param_name + ": " + str(param_value) + " " + param_unit
        
        new_prompt = """Evaluate the following premises:
[[Prompt]]

Used parameters: [[Params]].

Data points for used parameters: [[Data points]].

To identify the day as abnormal, it is enough that even one or more conditions are violated.
Remember to return at the end True if the day is abnormal or False otherwise.
"""

        new_prompt = new_prompt.replace("[[Prompt]]", prompt).replace("[[Params]]", used_params[:-2]).replace("[[Data points]]", param_values[:-2])

        new_dataset["Prompt"].append(new_prompt)
        new_dataset["Answer"].append(is_outlier)
                
    new_dataset_df = pd.DataFrame(new_dataset)
    new_dataset_df.to_csv("data/python_task2_llm_only_dataset_premises.csv", sep=";", columns=list(new_dataset.keys()), index=False)


generate_dataset_task2_llm()
# generate_dataset_task1_llm()
# generate_dataset_task2_llm()