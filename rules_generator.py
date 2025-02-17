import re
import numpy as np


# Function which generates In-context example depending on numbers of parameters in the input prompt
def get_python_rules(no_rules):
    # Read a ground truth rules for all 17 parameters
    f = open("templates/ground_truth_rules.py")
    gt_rules_lines = f.readlines()

    premises = []
    rules_lines = []
    rules = []

    # Retrieve lines of comments as input premises
    # And ruels functions
    for gt_rules_line in gt_rules_lines:
        if re.search("^# ", gt_rules_line):
            premises.append(gt_rules_line)
        else:
            rules_lines.append(gt_rules_line)

    # Merge both lines of rules into one string
    for i in range(0, len(rules_lines)-1, 3):
        rules.append(rules_lines[i] + rules_lines[i+1])

    # Prepare In-context examples: input premises and Lean4 rules
    input_premises = "Textual context: "
    input_rules = ""

    # print(rules)
    # Create a final rule for evaluation
    # def r5(x1: float, x2: float, x3: float, x4: float) -> bool:
    #     return not (r1(x1) and r2(x2) and r3(x3) and r4(x4))

    # general_rule_comment = "# To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    # general_rule = f"def r{no_rules+1}([args]) -> bool:\n   return not ([expr])"
    used_rules = ""
    used_args = ""
    expr = ""

    for i in range(no_rules):
        # Prepare rules, args, logical expression and premises
        # used_rules += f"r{i+1}("
        used_args += f"x{i+1}: float, "
        expr += f"r{i+1}(x{i+1}) and "
        input_premises += premises[i][2:]
        input_rules += premises[i] + rules[i] + "\n"

    # Cut the last 3 unused characters
    expr = expr[:-5]
    # used_args = used_args[:-2]
    # general_rule = general_rule.replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    # input_rules += general_rule_comment + general_rule

    f.close()

    return input_premises, input_rules


# Function which generates In-context example depending on numbers of parameters in the input prompt
def task2_get_python_rules(no_rules):
    # Read a ground truth rules for all 11 conditions
    f = open("templates/task2_ground_truth_rules.py")
    gt_rules_lines = f.readlines()
    f.close()

    premises = []
    rules_lines = []
    rules = []

    # Retrieve lines of comments as input premises
    # And rules functions
    for gt_rules_line in gt_rules_lines:
        if re.search("^# ", gt_rules_line):
            premises.append(gt_rules_line)
        else:
            rules_lines.append(gt_rules_line)

    # print("".join(rules_lines))
    # Merge both lines of rules into one string
    for i in range(0, len(rules_lines)-1, 6):
        rules.append("".join(rules_lines[i:i+5]))

    # print("\n##################################################################")
    # print(rules)
    # Prepare In-context examples: input premises and Lean4 rules
    input_premises = "Textual context: "
    input_rules = ""

    # print(rules)
    # Create a final rule for evaluation
    # def r5(x1: float, x2: float, x3: float, x4: float) -> bool:
    #     return not (r1(x1) and r2(x2) and r3(x3) and r4(x4))

    # general_rule_comment = "# To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    # general_rule = f"def r{no_rules+1}([args]) -> bool:\n   return not ([expr])"
    used_rules = ""
    used_args = ""
    expr = ""

    for i in range(no_rules):
        # Prepare rules, args, logical expression and premises
        # used_rules += f"r{i+1}("
        used_args += f"x{i+i+1}: float, "
        used_args += f"x{i+i+2}: float, "
        expr += f"r{i+1}(x{i+i+1} x{i+i+2}) and "
        input_premises += premises[i][2:]
        input_rules += premises[i] + rules[i] + "\n"

    # Cut the last 3 unused characters
    expr = expr[:-5]
    used_args = used_args[:-2]
    # general_rule = general_rule.replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    # input_rules += general_rule_comment + general_rule

    return input_premises, input_rules