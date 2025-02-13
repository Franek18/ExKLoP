import os
import torch
import argparse
import pandas as pd
import transformers

from rules_generator import get_python_rules

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Lllama-70", type=str, help="Which model to inference")
parser.add_argument("--index", default=0, type=int, help="At which index start the dataset inference")
args = parser.parse_args()


if args.model == "Mixtral":
    # Load Mixtral
    mixtral_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    mixtral = transformers.AutoModelForCausalLM.from_pretrained(mixtral_path, device_map="auto", torch_dtype=torch.bfloat16)
    mixtral_tokenizer = transformers.AutoTokenizer.from_pretrained(mixtral_path)

elif args.model == "Mistral":
    # Load Mistral
    mistral_path = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral = transformers.AutoModelForCausalLM.from_pretrained(mistral_path, device_map="auto", torch_dtype=torch.bfloat16)
    mistral_tokenizer = transformers.AutoTokenizer.from_pretrained(mistral_path)

elif args.model == "Qwen":
    # Load Qwen2-7B
    qwen2_path = "Qwen/Qwen2-7B-Instruct"
    qwen2 = transformers.AutoModelForCausalLM.from_pretrained(qwen2_path, device_map="auto", torch_dtype=torch.bfloat16)
    qwen2_tokenizer = transformers.AutoTokenizer.from_pretrained(qwen2_path)

elif args.model == "Llama-8":
    # Load Llama3-8B
    llama_8b_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama_8b_pipeline = transformers.pipeline(
        "text-generation",
        model=llama_8b_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    llama_8b_terminators = [
        llama_8b_pipeline.tokenizer.eos_token_id,
        llama_8b_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

elif args.model == "Llama-70":
    # Load Llama3-70B
    llama_70b_path = "meta-llama/Meta-Llama-3-70B-Instruct"

    llama_70b_pipeline = transformers.pipeline(
        "text-generation",
        model=llama_70b_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    llama_70b_terminators = [
        llama_70b_pipeline.tokenizer.eos_token_id,
        llama_70b_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

elif args.model == "Gemma":
        gemma_path = "google/gemma-1.1-7b-it"
        gemma_tokenizer = transformers.AutoTokenizer.from_pretrained(gemma_path)
        gemma = transformers.AutoModelForCausalLM.from_pretrained(gemma_path, device_map="auto", torch_dtype=torch.bfloat16)


# Load an In-context example of a rules defined in natural language
# f = open("templates/example_text.txt")
# example_text = f.read()
# f.close()

# # Load an Incontext example of a lean4 rules
# f = open("templates/example_lean.lean")
# example = f.read()
# f.close()

dataset_df = pd.read_csv("data/dataset_premises.csv", delimiter=";", header=0)

results = {"Prompt": [], args.model: []}

output_filename = f"outputs/Adapt_anonym_{args.model}_python_outputs.csv"

# current_outputs_df = pd.read_csv(f"outputs/{args.model}_outputs.csv", delimiter=";", header=0)

# current_len = len(current_outputs_df.index)

# A variable for current number of parameters in an input prompt
start_idx = args.index
curr_no_params = 0

for idx in dataset_df.index[start_idx:]:
    # A MistralAI template for a prompt
    mistralai_prompt = """[INST] System Message:
    You are a logician with a background in mathematics that translates natural language
    reasoning text to Python3 code so that these natural language reasoning problems can be solved.
    During the translation, please pay close attention to defining variables and rules.
    Do not add any comments from you.
    Be guided by the following example:
    Example input text:
    [[EXAMPLE_TEXT]]
    Example Python3 code:
    [[EXAMPLE]]
    [[INPUT]]
    [/INST]"""

    no_of_params = dataset_df["Number of parameters"][idx]

    # If we start next (greater) number of params than change In-context examples
    if curr_no_params < no_of_params:
        # Retrieve In-context examples
        example_text, example = get_python_rules(no_of_params)

        # A system prompt template for Llama3 and Qwen2
        system_prompt = """System Message:
        You are a logician with a background in mathematics that translates natural language
        reasoning text to Python3 code so that these natural language reasoning problems can be solved.
        During the translation, please pay close attention to defining variables and rules.
        Do not add any comments from you.
        Be guided by the following example:
        Example input text:
        [[EXAMPLE_TEXT]]
        Example Python3 code:
        [[EXAMPLE]]
        """

        # Modify input prompt
        system_prompt = system_prompt.replace('[[EXAMPLE_TEXT]]', example_text).replace('[[EXAMPLE]]', example)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
        ]

    curr_no_params = no_of_params
    
    mistralai_prompt = mistralai_prompt.replace('[[EXAMPLE_TEXT]]', example_text).replace('[[EXAMPLE]]', example)

    # print(messages[0]["content"])
    # Add Inconcext examples to the prompt
    
    # Get a prompt from a dataset
    prompt = dataset_df["Prompt"][idx] #+ "To identify the day as abnormal, it is enough that even one or more conditions are violated."

    # Prepare an input for Llama and Qwen models
    messages[1]["content"] = prompt

    full_prompt = system_prompt + "\n" + prompt

    if args.model == "Llama-8":
        llama_8b_prompt = llama_8b_pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate outputs for both Llama models 
        llama_8b_outputs = llama_8b_pipeline(
            llama_8b_prompt,
            max_new_tokens=2048,
            eos_token_id=llama_8b_terminators,
            temperature=0,
            do_sample=False
        )


        output = llama_8b_outputs[0]["generated_text"].split("<|end_header_id|>")[-1]

    elif args.model == "Llama-70":
        llama_70b_prompt = llama_70b_pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
    
        llama_70b_outputs = llama_70b_pipeline(
            llama_70b_prompt,
            max_new_tokens=2048,
            eos_token_id=llama_70b_terminators,
            temperature=0,
            do_sample=False
        )


        output = llama_70b_outputs[0]["generated_text"].split("<|end_header_id|>")[-1]

    elif args.model == "Qwen":
        qwen2_prompt = qwen2_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )

        qwen2_inputs = qwen2_tokenizer([qwen2_prompt], return_tensors="pt")
        # Generate outputs for Qwen model
        qwen2_generated_ids = qwen2.generate(**qwen2_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
        qwen2_output = qwen2_tokenizer.batch_decode(qwen2_generated_ids)[0]
        output = qwen2_output.replace('<|im_end|>', "").split("<|im_start|>assistant")[-1]


    elif args.model == "Mistral":
        # Prepare inputs for MistralAi models
        mistralai_prompt = mistralai_prompt.replace('[[INPUT]]', prompt)

        # print(mistralai_prompt)
        # print("-------------------------------------------")

        mistral_inputs = mistral_tokenizer([mistralai_prompt], return_tensors="pt")
        mistral_generated_ids = mistral.generate(**mistral_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
        output = mistral_tokenizer.batch_decode(mistral_generated_ids)[0]
        output = output.split("[/INST]")[-1].replace("</s>", "")


    elif args.model == "Mixtral":
        # Prepare inputs for MistralAi models
        mistralai_prompt = mistralai_prompt.replace('[[INPUT]]', prompt)

        mixtral_inputs = mixtral_tokenizer([mistralai_prompt], return_tensors="pt")

        mixtral_generated_ids = mixtral.generate(**mixtral_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
        output = mixtral_tokenizer.batch_decode(mixtral_generated_ids)[0]
        output = output.split("[/INST]")[-1].replace("</s>", "")


    elif args.model == "Gemma":
        gemma_prompt = system_prompt + "\n" + prompt
        # print(gemma_prompt + "\n")
        prompt_len = len(gemma_prompt) + 5 # 5 is a length of output <bos> token
        input_ids = gemma_tokenizer(gemma_prompt, return_tensors="pt")

        output = gemma.generate(**input_ids, max_new_tokens=2048, temperature=0, do_sample=False)
        output = gemma_tokenizer.decode(output[0])[prompt_len:]


    results["Prompt"].append(full_prompt)
    results[args.model].append(output)

    if idx % 50 == 0:
        print(f"Idx == {idx} - saving batch of data")
        result_pd = pd.DataFrame(results)
        if not os.path.exists(output_filename):
            result_pd.to_csv(output_filename, sep=";", columns=list(results.keys()), index=False)
        else:
            result_pd.to_csv(output_filename, sep=";", mode='a', index=False, header=False)

        results["Prompt"] = []
        results[args.model] = []

result_pd = pd.DataFrame(results)
result_pd.to_csv(output_filename, sep=";", mode='a', index=False, header=False)