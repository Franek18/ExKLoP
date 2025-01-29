import re
import numpy as np

def get_rules(no_rules):
    # Read a ground truth rules for all 17 parameters
    f = open("templates/ground_truth_rules.lean")
    gt_rules_lines = f.readlines()

    premises = []
    rules_lines = []
    rules = []

    # Retrieve lines of comments as input premises
    # And ruels functions
    for gt_rules_line in gt_rules_lines:
        if re.search("^-- ", gt_rules_line):
            premises.append(gt_rules_line)
        else:
            rules_lines.append(gt_rules_line)

    # Merge both lines of rules into one string
    for i in range(0, len(rules_lines)-1, 2):
        rules.append(rules_lines[i] + rules_lines[i+1])

    # Prepare In-context examples: input premises and Lean4 rules
    input_premises = "Textual context: "
    input_rules = ""

    # Create a final rule for evaluation
    # def r5 (r1 r2 r3 r4 : Float -> Bool) (x1 x2 x3 x4 : Float) : Bool :=
    #   ¬((r1 x1) ∧ (r2 x2) ∧ (r3 x3) ∧ (r4 x4))
    general_rule_comment = "-- To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    general_rule = f"def r{no_rules+1} ([rules]: Float -> Bool) ([args]: Float) : Bool :=\n  ¬([expr])"
    used_rules = ""
    used_args = ""
    expr = ""

    for i in range(no_rules):
        # Prepare rules, args, logical expression and premises
        used_rules += f"r{i+1} "
        used_args += f"x{i+1} "
        expr += f"(r{i+1} x{i+1}) ∧ "
        input_premises += premises[i][3:]
        input_rules += premises[i] + rules[i]

    # Cut the last 3 unused characters
    expr = expr[:-3]
    general_rule = general_rule.replace("[rules]", used_rules).replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    input_rules += "\n" + general_rule_comment + general_rule


    # for rule in rules:
    #     print(rule)
    # for premise in premises:
    #     print(premise)

    # print(input_rules)
    # print(input_premises)

    f.close()

    return example_prompt

# Function which generates In-context example depending on numbers of parameters in the input prompt
def get_rules(no_rules):
    # Read a ground truth rules for all 17 parameters
    f = open("templates/ground_truth_rules.lean")
    gt_rules_lines = f.readlines()

    premises = []
    rules_lines = []
    rules = []

    # Retrieve lines of comments as input premises
    # And ruels functions
    for gt_rules_line in gt_rules_lines:
        if re.search("^-- ", gt_rules_line):
            premises.append(gt_rules_line)
        else:
            rules_lines.append(gt_rules_line)

    # Merge both lines of rules into one string
    for i in range(0, len(rules_lines)-1, 2):
        rules.append(rules_lines[i] + rules_lines[i+1])

    # Prepare In-context examples: input premises and Lean4 rules
    input_premises = "Textual context: "
    input_rules = ""

    # Create a final rule for evaluation
    # def r5 (r1 r2 r3 r4 : Float -> Bool) (x1 x2 x3 x4 : Float) : Bool :=
    #   ¬((r1 x1) ∧ (r2 x2) ∧ (r3 x3) ∧ (r4 x4))
    general_rule_comment = "-- To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    general_rule = f"def r{no_rules+1} ([rules]: Float -> Bool) ([args]: Float) : Bool :=\n  ¬([expr])"
    used_rules = ""
    used_args = ""
    expr = ""

    for i in range(no_rules):
        # Prepare rules, args, logical expression and premises
        used_rules += f"r{i+1} "
        used_args += f"x{i+1} "
        expr += f"(r{i+1} x{i+1}) ∧ "
        input_premises += premises[i][3:]
        input_rules += premises[i] + rules[i]

    # Cut the last 3 unused characters
    expr = expr[:-3]
    general_rule = general_rule.replace("[rules]", used_rules).replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    input_rules += "\n" + general_rule_comment + general_rule


    # for rule in rules:
    #     print(rule)
    # for premise in premises:
    #     print(premise)

    # print(input_rules)
    # print(input_premises)

    f.close()

    return input_premises, input_rules


def task2_get_rules(no_rules):
    # Read a ground truth rules for all 21 statements
    f = open("templates/task2_ground_truth_rules.lean")
    gt_rules_lines = f.readlines()

    premises = []
    rules_lines = []
    rules = []

    # Retrieve lines of comments as input premises
    # And ruels functions
    for gt_rules_line in gt_rules_lines:
        if re.search("^-- ", gt_rules_line):
            premises.append(gt_rules_line)
        else:
            rules_lines.append(gt_rules_line)

    

    # Merge both lines of rules into one string
    for i in range(0, len(rules_lines)-1, 2):
        rules.append(rules_lines[i] + rules_lines[i+1])

    # Prepare In-context examples: input premises and Lean4 rules
    input_premises = "Textual context: "
    input_rules = ""

    # Create a final rule for evaluation
    # def r3 (r1 r2 : Float -> Bool) (x1 x2 x3 x4 : Float) : Bool :=
    #     ¬((r1 x1 x2) ∧ (r2 x3 x4))
    general_rule_comment = "-- To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    general_rule = f"def r{no_rules+1} ([args]: Float) : Bool :=\n  ¬([expr])"
    used_rules = ""
    used_args = ""
    expr = ""

    # Add no_rules - 1 rules and leave space for the last rule with 4 parameters
    for i in range(no_rules-1):
        # Prepare rules, args, logical expression and premises
        used_args += f"x{i+i+1} "
        used_args += f"x{i+i+2} "
        expr += f"(r{i+1} x{i+i+1} x{i+i+2}) ∧ "
        input_premises += premises[i][3:]
        input_rules += premises[i] + rules[i]

    # Add the last rule, that with 4 parameters
    used_args += f"x{no_rules-1+no_rules-1+1} "
    used_args += f"x{no_rules-1+no_rules-1+2} "
    used_args += f"x{no_rules-1+no_rules-1+3} "
    used_args += f"x{no_rules-1+no_rules-1+4} "
    
    expr += f"(r{no_rules} x{no_rules-1+no_rules-1+1} x{no_rules-1+no_rules-1+2} x{no_rules-1+no_rules-1+3} x{no_rules-1+no_rules-1+4}) ∧ "

    input_premises += premises[-1][3:]
    # Change the r21 to number of the last rule
    input_rules += premises[-1] + rules[-1].replace("r21", f"r{no_rules}")
    # Cut the last 3 unused characters
    expr = expr[:-3]
    general_rule = general_rule.replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    input_rules += "\n" + general_rule_comment + general_rule


    # for rule in rules:
    #     print(rule)
    # for premise in premises:
    #     print(premise)

    # print(input_rules)
    # print(input_premises)

    f.close()

    return input_premises, input_rules


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

    general_rule_comment = "# To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    general_rule = f"def r{no_rules+1}([args]) -> bool:\n   return not ([expr])"
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
    used_args = used_args[:-2]
    general_rule = general_rule.replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    input_rules += general_rule_comment + general_rule

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

    general_rule_comment = "# To identify the day as abnormal, it is enough that even one or more conditions are violated.\n"
    general_rule = f"def r{no_rules+1}([args]) -> bool:\n   return not ([expr])"
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
    general_rule = general_rule.replace("[args]", used_args).replace("[expr]", expr)

    # Add general rule code to the whole Lean4 code
    # input_rules += general_rule_comment + general_rule

    return input_premises, input_rules