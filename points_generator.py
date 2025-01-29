import re
import ast
import random
import numpy as np
import pandas as pd


def generate_points():
    parameters_df = pd.read_csv("data/parameters.csv", delimiter=";", header=0)
    points_dataset = {"Parameter": [], "Value": [], "Unit": [], "Outlier": []}

    # Number of values for each parameter
    p = 50

    for idx in parameters_df.index:
        parameter = parameters_df["Parameters"][idx]
        param_min = parameters_df["Min"][idx]
        param_max = parameters_df["Max"][idx]
        param_unit = parameters_df["Unit"][idx]

        # For each parameter generate p values 
        if param_unit == "-":
            # use randint for integer values
            # param_values = np.random.randint(size = p, low = param_min - int(0.1*param_min), high = param_max + int(0.1*param_min))
            low_left_param_value = np.random.randint(size = 1, low = param_min - int(0.1*param_min), high = param_min - int(0.05*param_min))
            low_right_param_value = np.random.randint(size = 1, low = param_min + int(0.05*param_min), high = param_min + int(0.1*param_min))

            high_left_param_value = np.random.randint(size = 1, low = param_max - int(0.1*param_max), high = param_max - int(0.05*param_max))
            high_right_param_value = np.random.randint(size = 1, low = param_max + int(0.05*param_max), high = param_max + int(0.1*param_max))
        else:
            # for the rest use uniform distribution
            # param_values = np.random.uniform(size = p, low = param_min - 0.1*param_min, high = param_max + 0.1*param_min)
            low_left_param_value = np.random.uniform(size = 1, low = param_min - 0.1*param_min, high = param_min - 0.05*param_min)
            low_right_param_value = np.random.uniform(size = 1, low = param_min + 0.05*param_min, high = param_min + 0.1*param_min)

            high_left_param_value = np.random.uniform(size = 1, low = param_max - 0.1*param_max, high = param_max - 0.05*param_max)
            high_right_param_value = np.random.uniform(size = 1, low = param_max + 0.05*param_max, high = param_max + 0.1*param_max)         

        for param_value in [low_left_param_value, low_right_param_value, high_left_param_value, high_right_param_value]:
            points_dataset["Parameter"].append(parameter)
            points_dataset["Value"].append(round(param_value[0], 2))
            points_dataset["Unit"].append(param_unit)
            if param_value >= param_min and param_value <= param_max:
                points_dataset["Outlier"].append(0)
            else:
                points_dataset["Outlier"].append(1)


    points_dataset_df = pd.DataFrame(points_dataset)
    points_dataset_df.to_csv("data/points_dataset.csv", sep=";", columns=list(points_dataset.keys()), index=False)


def old_task2_generate_points():
    statements_df = pd.read_csv("data/task2_statements.csv", delimiter=";", header=0)
    dataset_df = pd.read_csv("data/task2_dataset_premises.csv", delimiter=";", header=0)
    parameters_df = pd.read_csv("data/parameters.csv", delimiter=";", header=0)

    points_dataset = {"Premise_number": [], "Values": [], "Outlier": []}

    # Number of values for each parameter
    p = 50

    for idx in dataset_df.index:
        # Remove redundant text from a prompt
        prompt = dataset_df["Prompt"][idx].replace("Input text:\nTextual context: ", "")
        # statement_no = dataset_df["Number of statements"][idx]
        all_params = ast.literal_eval(dataset_df["Parameters"][idx])

        # Take all premises, except the last one about detecting abnormal day
        prompt_premises = prompt.split("\n")[:-1]
        
        values = []
        is_outlier = 0
        # print(prompt_premises)
        # Iterate over set of parameters used in a prompt
        for params, premise in zip(all_params, prompt_premises):
            # Transform an array of parameters into a string which identifies statement in statements_df
            params_string = params[0]
            for param in params[1:]:
                params_string += ", " + param

            # Just in case remove from a premise namex of parameters with numbers
            premise = premise.replace("axle 1", "").replace("axle 2", "").replace("axle 3", "")
            # Retrieve values of parameters that make constraints
            params_values = re.findall("[0-9]+[.]*[0-9]*", premise)

            # Get statement record from statements_df
            statement_idx = statements_df[statements_df["Parameters"] == params_string].index[0]
            # print(statement_idx)

            # And retrieve relation of each parameter in a statement
            param_relations = statements_df["Relation"][statement_idx].split(",")
            # print(param_relations)
            # Array for params relations in a statement
            rel_arr = []
            for relation in param_relations:
                # Retrieve relation i.e. relation = "x == X" -> relation.split(" ") -> ["x", "==", "X"] -> rel[1] = "=="
                rel = relation.split(" ")[1]
                rel_arr.append(rel)
            
            
            for param, cap_param_value, rel in zip(params, params_values, rel_arr):
                # Retrieve record of given parameter
                param_idx = parameters_df[parameters_df["Parameters"] == param].index[0]
                # Cast string to float
                cap_param_value = float(cap_param_value)

                param_min = parameters_df["Min"][param_idx]
                param_max = parameters_df["Max"][param_idx]
                param_unit = parameters_df["Unit"][param_idx]

                # case = random.choice([0, 1])
                case = random.random()

                # print(f"Param: {param}, Constrain value: {cap_param_value}, min value: {param_min}, max_value: {param_max}")
                if case < 0.0:
                    is_outlier = 1
                    # Param value isn't consistent to constraint
                    # Generate value for a parameter
                    if rel == "==":
                        param_value = cap_param_value
                        if param_unit == "-":
                            # use randint for integer values
                            while param_value == cap_param_value:
                                param_value = np.random.randint(low = param_min, high = param_max)
                        else:
                            # for the rest use uniform distribution
                            while param_value == cap_param_value:
                                param_value = round(np.random.uniform(low = param_min, high = param_max), 1)
                    elif rel == ">":
                        param_value = cap_param_value + 1
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value > cap_param_value:
                            param_value = np.random.randint(low = param_min, high = cap_param_value)
                        else:
                            # for the rest use uniform distribution
                            # while param_value > cap_param_value:
                            param_value = round(np.random.uniform(low = param_min, high = cap_param_value), 1)
                    elif rel == "<":
                        param_value = cap_param_value - 1
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value <= cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = np.random.randint(low = param_max, high = cap_param_value)
                            else:
                                param_value = np.random.randint(low = cap_param_value, high = param_max)
                        else:
                            # for the rest use uniform distribution
                            # while param_value <= cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = round(np.random.uniform(low = param_max, high = cap_param_value), 1)
                            else:
                                param_value = round(np.random.uniform(low = cap_param_value, high = param_max), 1)
                        param_value = cap_param_value
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value >= cap_param_value:
                            param_value = np.random.randint(low = param_min, high = cap_param_value)
                        else:
                            # for the rest use uniform distribution
                            # while param_value >= cap_param_value:
                            param_value = round(np.random.uniform(low = param_min, high = cap_param_value), 1)                        
                    elif rel == "<=":
                        param_value = cap_param_value
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value <= cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = np.random.randint(low = param_max, high = cap_param_value)
                            else:
                                param_value = np.random.randint(low = cap_param_value, high = param_max)
                        else:
                            # for the rest use uniform distribution
                            # while param_value <= cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = round(np.random.uniform(low = param_max, high = cap_param_value), 1)
                            else:
                                param_value = round(np.random.uniform(low = cap_param_value, high = param_max), 1)
                else:
                    # Param value is consistent to constraint
                    if rel == "==":
                        param_value = cap_param_value
                    elif rel == "<=":
                        param_value = cap_param_value + 1
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value > cap_param_value:
                            param_value = np.random.randint(low = param_min, high = cap_param_value)
                        else:
                            # for the rest use uniform distribution
                            # while param_value > cap_param_value:
                            param_value = round(np.random.uniform(low = param_min, high = cap_param_value), 1)
                    elif rel == "<":
                        param_value = cap_param_value + 1
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value > cap_param_value:
                            param_value = np.random.randint(low = param_min, high = cap_param_value - 1)
                        else:
                            # for the rest use uniform distribution
                            # while param_value > cap_param_value:
                            param_value = round(np.random.uniform(low = param_min, high = cap_param_value - 1), 1)
                    elif rel == ">":
                        param_value = cap_param_value - 1
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value < cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = np.random.randint(low = param_min, high = param_max)
                            else:
                                param_value = np.random.randint(low = cap_param_value + 1, high = param_max)
                        else:
                            # for the rest use uniform distribution
                            # while param_value <= cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = round(np.random.uniform(low = param_min, high = param_max), 1)
                            else:
                                param_value = round(np.random.uniform(low = cap_param_value + 1, high = param_max), 1)                      
                    elif rel == ">=":
                        param_value = cap_param_value - 1
                        if param_unit == "-":
                            # use randint for integer values
                            # while param_value < cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = np.random.randint(low = param_min, high = param_max)
                            else:
                                param_value = np.random.randint(low = cap_param_value, high = param_max)
                        else:
                            # for the rest use uniform distribution
                            # while param_value <= cap_param_value:
                            if cap_param_value >= param_max:
                                param_value = round(np.random.uniform(low = param_min, high = param_max), 1)
                            else:
                                param_value = round(np.random.uniform(low = cap_param_value, high = param_max), 1)                      

                values.append(param_value)

        points_dataset["Premise_number"].append(idx)
        points_dataset["Values"].append(values)
        points_dataset["Outlier"].append(is_outlier)


    points_dataset_df = pd.DataFrame(points_dataset)
    points_dataset_df.to_csv("data/old_task2_points_dataset.csv", sep=";", columns=list(points_dataset.keys()), index=False)


def prob_values2_5(params_unit, params_min_limits, params_max_limits):
    values = {"Greater": [], "Lower": [], "Equal": []}
    # First param > second param
    if params_unit == "-":
        # use randint for integer values
        first_param_value = np.random.randint(low = params_min_limits[0] + 1, high = params_max_limits[0])
        second_param_value = np.random.randint(low = params_min_limits[0], high = first_param_value - 1)
    else:
        # for the rest use uniform distribution
        first_param_value = round(np.random.uniform(low = params_min_limits[0] + 1, high = params_max_limits[0]), 1)
        second_param_value = round(np.random.uniform(low = params_min_limits[0], high = first_param_value - 1), 1)                       

    values["Greater"].extend([first_param_value, second_param_value])

    # First param < second param 
    if params_unit == "-":
        # use randint for integer values
        second_param_value = np.random.randint(low = params_min_limits[1] + 1, high = params_max_limits[1])
        first_param_value = np.random.randint(low = params_min_limits[1], high = second_param_value - 1)
    else:
        # for the rest use uniform distribution
        second_param_value = round(np.random.uniform(low = params_min_limits[1] + 1, high = params_max_limits[1]), 1)
        first_param_value = round(np.random.uniform(low = params_min_limits[1], high = second_param_value - 1), 1)                       

    values["Lower"].extend([first_param_value, second_param_value])

    # First param == second param 
    if params_unit == "-":
        # use randint for integer values
        first_param_value = np.random.randint(low = params_min_limits[0] + 1, high = params_max_limits[0])
        second_param_value = first_param_value
    else:
        # for the rest use uniform distribution
        first_param_value = round(np.random.uniform(low = params_min_limits[0] + 1, high = params_max_limits[0]), 1)
        second_param_value = first_param_value

    values["Equal"].extend([first_param_value, second_param_value])

    return values


def prob_values6_10(params_unit, params_min_limits, params_max_limits):
    values = {"Greater": [], "Lower": [], "Equal": []}
    # First param > second param
    if params_unit == "-":
        # use randint for integer values
        first_param_value = np.random.randint(low = params_min_limits[0] + 1, high = params_max_limits[0])
        second_param_value = np.random.randint(low = params_min_limits[0], high = first_param_value - 1)
    else:
        # for the rest use uniform distribution
        first_param_value = round(np.random.uniform(low = params_min_limits[0] + 1, high = params_max_limits[0]), 1)
        second_param_value = round(np.random.uniform(low = params_min_limits[0], high = first_param_value - 1), 1)                       

    values["Greater"].extend([first_param_value, second_param_value])

    # First param < second param 
    if params_unit == "-":
        # use randint for integer values
        second_param_value = np.random.randint(low = params_min_limits[1] + 1, high = params_max_limits[1])
        first_param_value = np.random.randint(low = params_min_limits[1], high = second_param_value - 1)
    else:
        # for the rest use uniform distribution
        second_param_value = round(np.random.uniform(low = params_min_limits[1] + 1, high = params_max_limits[1]), 1)
        first_param_value = round(np.random.uniform(low = params_min_limits[1], high = second_param_value - 1), 1)                       

    values["Lower"].extend([first_param_value, second_param_value])

    # First param == second param 
    if params_unit == "-":
        # use randint for integer values
        first_param_value = np.random.randint(low = params_min_limits[0] + 1, high = params_max_limits[0])
        second_param_value = first_param_value
    else:
        # for the rest use uniform distribution
        first_param_value = round(np.random.uniform(low = params_min_limits[0] + 1, high = params_max_limits[0]), 1)
        second_param_value = first_param_value

    values["Equal"].extend([first_param_value, second_param_value])

    return values


def prob_values_cond11(params_min_limits, params_max_limits):
    values = {"Greater": [], "Lower": [], "Equal": []}
    # First param > second param
    first_param_value = round(np.random.uniform(low = params_min_limits[0], high = params_max_limits[0]), 1)
    second_param_value = round(np.random.uniform(low = params_min_limits[1], high = first_param_value/3 - 0.1), 1)                       
    third_param_value = round(np.random.uniform(low = params_min_limits[2], high = first_param_value/3), 1)
    fourth_param_value = round(np.random.uniform(low = params_min_limits[3], high = first_param_value/3), 1)     

    values["Greater"].extend([first_param_value, second_param_value, third_param_value, fourth_param_value])

    # First param < second param 
    second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)                       
    third_param_value = round(np.random.uniform(low = params_min_limits[2], high = params_max_limits[2]), 1)
    fourth_param_value = round(np.random.uniform(low = params_min_limits[3], high = params_max_limits[3]), 1)

    sum_params = second_param_value + third_param_value + fourth_param_value

    first_param_value = round(np.random.uniform(low = params_min_limits[0], high = min(params_max_limits[0], sum_params)), 1)                 

    values["Lower"].extend([first_param_value, second_param_value, third_param_value, fourth_param_value])

    # First param == second param 
    second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[0] / 3), 1)                       
    third_param_value = round(np.random.uniform(low = params_min_limits[2], high = params_max_limits[0] / 3), 1)
    fourth_param_value = round(np.random.uniform(low = params_min_limits[3], high = params_max_limits[0] / 3), 1)

    first_param_value = second_param_value + third_param_value + fourth_param_value

    values["Equal"].extend([first_param_value, second_param_value, third_param_value, fourth_param_value])

    return values


def prob_values_cond1_2(condition_no, params_min_limits, params_max_limits):
    values = {"Greater": [], "Lower": [], "Equal": []}
    if condition_no == 0:
        # Second param / first param > 0.25
        second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)
        first_param_value = 2 * second_param_value

        values["Greater"].extend([first_param_value, second_param_value])
        # Second param / first param == 0.25
        second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)
        first_param_value = 4 * second_param_value

        values["Equal"].extend([first_param_value, second_param_value])

        # Second param / first param < 0.25
        second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)
        first_param_value = 10 * second_param_value

        values["Lower"].extend([first_param_value, second_param_value])

    elif condition_no == 1:
        # Second param / first param > 2
        second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)
        first_param_value = 0.3 * second_param_value

        values["Greater"].extend([first_param_value, second_param_value])
        # Second param / first param == 2
        second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)
        first_param_value = 0.5 * second_param_value

        values["Equal"].extend([first_param_value, second_param_value])

        # Second param / first param < 2
        second_param_value = round(np.random.uniform(low = params_min_limits[1], high = params_max_limits[1]), 1)
        first_param_value = 1.5 * second_param_value

        values["Lower"].extend([first_param_value, second_param_value])

    return values


def task2_generate_points():
    conditions_df = pd.read_csv("data/Task2_conditions_updated.csv", delimiter=",", header=0)
    parameters_df = pd.read_csv("data/parameters.csv", delimiter=";", header=0)
    conditions_df["Parameters_deafult"] = conditions_df["Parameters"].apply(lambda x: x.split("] ")[0])

    points_dataset = {"Condition_number": [], "Values": [], "Outlier": []}
    # Iterate over set of parameters used in a prompt
    for idx in conditions_df.index:
        is_outlier = 0
        # Transform an array of parameters into a string which identifies statement in statements_df
        params = conditions_df["Parameters_deafult"][idx].split(", ")
            
        first_param_idx = parameters_df[parameters_df["Parameters"] == params[0]].index[0]
        second_param_idx = parameters_df[parameters_df["Parameters"] == params[1]].index[0]

        first_param_min = parameters_df["Min"][first_param_idx]
        first_param_max = parameters_df["Max"][first_param_idx]
        first_param_unit = parameters_df["Unit"][first_param_idx]
        second_param_unit = parameters_df["Unit"][second_param_idx]

        second_param_min = parameters_df["Min"][second_param_idx]
        second_param_max = parameters_df["Max"][second_param_idx]       

        if idx < 2:
            # Numbered params
            params_values = prob_values_cond1_2(idx, [first_param_min, second_param_min], [first_param_max, second_param_max])
            
            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Greater"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Equal"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Lower"])
            points_dataset["Outlier"].append(1)            
        elif idx < 6:
            # First param >= second param
            params_values = prob_values2_5(first_param_unit, [first_param_min, second_param_min], [first_param_max, second_param_max])

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Greater"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Equal"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Lower"])
            points_dataset["Outlier"].append(1)             
        elif idx < 10:
            # First param <= second param
            params_values = prob_values6_10(first_param_unit, [first_param_min, second_param_min], [first_param_max, second_param_max])

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Greater"])
            points_dataset["Outlier"].append(1)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Equal"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Lower"])
            points_dataset["Outlier"].append(0) 
        elif idx == 10:
            # First param + second param + third param <= fourth param
            third_param_idx = parameters_df[parameters_df["Parameters"] == params[2]].index[0]
            fourth_param_idx = parameters_df[parameters_df["Parameters"] == params[3]].index[0]

            third_param_min = parameters_df["Min"][third_param_idx]
            third_param_max = parameters_df["Max"][third_param_idx]

            fourth_param_min = parameters_df["Min"][fourth_param_idx]
            fourth_param_max = parameters_df["Max"][fourth_param_idx]

            params_values = prob_values_cond11([first_param_min, second_param_min, third_param_min, fourth_param_min], [first_param_max, second_param_max, third_param_max, fourth_param_max])
                
            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Greater"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Equal"])
            points_dataset["Outlier"].append(0)

            points_dataset["Condition_number"].append(idx)
            points_dataset["Values"].append(params_values["Lower"])
            points_dataset["Outlier"].append(1) 


    points_dataset_df = pd.DataFrame(points_dataset)
    points_dataset_df.to_csv("data/task2_points_dataset.csv", sep=";", columns=list(points_dataset.keys()), index=False)


# generate_points()
# detect_outlier_point()
task2_generate_points()
