import os
import re
import ast
import copy
import torch
import argparse
import warnings
import pandas as pd
import transformers

from rules_generator import get_python_rules

warnings.simplefilter('ignore')
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default="Lllama-70", type=str, help="Which model to inference")
# parser.add_argument("--refinement", default="syntax", type=str, help="Which method of refinement- syntax|runtime|logic")
# parser.add_argument("--step", default=1, type=int, help="Which step of refinement 1-10")
# parser.add_argument("--index", default=0, type=int, help="At which index start the dataset inference")
# args = parser.parse_args()

def task1_refinement(model, refinement, step, index=0):
    if model == "Gemma3":
        # Load Gemma-12
        gemma_12b_pipeline = transformers.pipeline(
            "text-generation",               # "image-text-to-text",
            model="google/gemma-3-12b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
    elif model == "Llama-8":
        # Load Llama3-8B
        llama_8b_path = "meta-llama/Llama-3.1-8B-Instruct"
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

    elif model == "Llama-70":
        # Load Llama3-70B
        llama_70b_path = "meta-llama/Llama-3.3-70B-Instruct"

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

    elif model == "QwenCoder":
        # Load Qwen2.5-Coder-14B-Instruc
        qwencoder_path = "Qwen/Qwen2.5-Coder-14B-Instruct"
        qwencoder = transformers.AutoModelForCausalLM.from_pretrained(qwencoder_path, device_map="auto", torch_dtype=torch.bfloat16)
        qwencoder_tokenizer = transformers.AutoTokenizer.from_pretrained(qwencoder_path)

    elif model == "QwenCoderMedium":
        # Load Qwen2.5-Coder-14B-Instruc
        qwencodermedium_path = "Qwen/Qwen2.5-Coder-3B-Instruct"
        qwencodermedium = transformers.AutoModelForCausalLM.from_pretrained(qwencodermedium_path, device_map="auto", torch_dtype=torch.bfloat16)
        qwencodermedium_tokenizer = transformers.AutoTokenizer.from_pretrained(qwencodermedium_path)
        
    elif model == "QwenCoderSmall":
        # Load Qwen2.5-Coder-14B-Instruc
        qwencodersmall_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        qwencodersmall = transformers.AutoModelForCausalLM.from_pretrained(qwencodersmall_path, device_map="auto", torch_dtype=torch.bfloat16)
        qwencodersmall_tokenizer = transformers.AutoTokenizer.from_pretrained(qwencodersmall_path)

    elif model == "Codestral":
        # Load Mistral
        codestral_path = "mistralai/Codestral-22B-v0.1"
        codestral = transformers.AutoModelForCausalLM.from_pretrained(codestral_path, device_map="auto", torch_dtype=torch.bfloat16)
        codestral_tokenizer = transformers.AutoTokenizer.from_pretrained(codestral_path)

    if refinement == "syntax":
        if step == 0:
            dataset_df = pd.read_csv(f"results/first/Task1/{model}_first_step_0_results.csv", delimiter=";", header=0)
        else:
            dataset_df = pd.read_csv(f"results/logic/Task1/{model}_logic_step_{step-1}_results.csv", delimiter=";", header=0)
        results = {"Prompt": [], model: []}
        output_filename = f"outputs/Task1/{model}_step_{step}_syntax_outputs.csv"

        system_prompt_filename = "templates/prompts/task1_syntax_system_prompt.txt"

    elif refinement == "runtime":
        dataset_df = pd.read_csv(f"results/syntax/Task1/{model}_syntax_step_{step}_results.csv", delimiter=";", header=0)
        results = {"Prompt": [], model: []}
        output_filename = f"outputs/Task1/{model}_step_{step}_runtime_outputs.csv"

        system_prompt_filename = "templates/prompts/task1_runtime_system_prompt.txt"

    elif refinement == "logic":
        dataset_df = pd.read_csv(f"results/runtime/Task1/{model}_runtime_step_{step}_results.csv", delimiter=";", header=0)
        results = {"Prompt": [], model: []}
        output_filename = f"outputs/Task1/{model}_step_{step}_logic_outputs.csv"

        system_prompt_filename = "templates/prompts/task1_logic_system_prompt.txt"

    # dataset_df = pd.read_csv(output_dataset, delimiter=";", header=0)
    updated_dataset_df = copy.deepcopy(dataset_df)
    # A variable for current number of parameters in an input prompt
    start_idx = index
    curr_no_params = 0

    for idx in dataset_df.index[start_idx:]:

        if refinement == "syntax":
            syntax_answers = dataset_df["Syntax eval"][idx]
            if syntax_answers != "Correct syntax":
                error_message = "Syntax evaluation returns the following error:\n" + syntax_answers
            else:
                continue

        elif refinement == "runtime":
            gt_answers = dataset_df["Outlier"][idx]
            model_answers = dataset_df["Outlier detection"][idx]

            if gt_answers == "Error":
                error_message = "Code after execution returns the following error:\n" + model_answers
            else:
                continue

        elif refinement == "logic":
            gt_answers = dataset_df["Outlier"][idx]
        
            # Check whether GT answer indicates runtime error, if yes, than skip
            if gt_answers == "Error":
                continue

            # Retrieve GT answer, model answer and model Python rules
            gt_answers = ast.literal_eval(dataset_df["Outlier"][idx])
            model_answers = ast.literal_eval(dataset_df["Outlier detection"][idx])
            python_output = dataset_df["Model output"][idx]

            # Retrieve the implemantation of all rules (without the final rule) from model's output
            all_rules = re.findall(r"def .+?:\n\s+return .+", python_output)

            error_message = "Model failed with generation of the proper logic for the following rules:\n"
            wrong_rules = []
            for model_answer, gt_answer in zip(model_answers, gt_answers):
                    model_answer = model_answer.split(", ")
                    gt_answer = gt_answer.split(", ")

                    # Iterate over answers for each rule, and for each rule
                    for model_rule_answer, gt_rule_answer, rule in zip(model_answer[1:], gt_answer[1:], all_rules):
                        if model_rule_answer != gt_rule_answer:
                            # If answers are different this means that this rule has wrong implementation
                            wrong_rules.append(rule)

            if len(wrong_rules) == 0:
                # Check there are any wrong premises, if not go to next input prompt
                continue

            # else prepare list of deduplicated wrong premises
            wrong_rules = set(wrong_rules)
            error_message += "\n".join(wrong_rules)        
    
        print(f"Replacing output no. {idx}")

        f = open(system_prompt_filename)
        prompt = f.read()
        f.close()
        # A MistralAI template for a prompt
        mistralai_prompt = f"[INST]{prompt}[/INST]"

        wrong_python_code = dataset_df["Model output"][idx]
        input_text = dataset_df["Prompt"][idx].split('Textual context: ')[-1]
        
        mistralai_prompt = mistralai_prompt.replace('[[PYTHON3 CODE]]', wrong_python_code).replace('[[INPUT TEXT]]', input_text).replace('[[ERROR]]', error_message)
        prompt = prompt.replace('[[PYTHON3 CODE]]', wrong_python_code).replace('[[INPUT TEXT]]', input_text).replace('[[ERROR]]', error_message)

        # Prepare an input for Llama and Qwen models
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        if model == "Llama-8":
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
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )


            output = llama_8b_outputs[0]["generated_text"].split("<|end_header_id|>")[-1]

        elif model == "Llama-70":
            llama_70b_prompt = llama_70b_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
        
            llama_70b_outputs = llama_70b_pipeline(
                llama_70b_prompt,
                max_new_tokens=2048,
                eos_token_id=llama_70b_terminators,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )


            output = llama_70b_outputs[0]["generated_text"].split("<|end_header_id|>")[-1]

        elif model == "Gemma3":
            # print(gemma_prompt + "\n")
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": ""}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            output = gemma_12b_pipeline(
                text_inputs=messages,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )

            output = output[0]["generated_text"][-1]["content"]

        elif model == "QwenCoder":
            qwencoder_prompt = qwencoder_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )

            qwencoder_inputs = qwencoder_tokenizer([qwencoder_prompt], return_tensors="pt")
            # Generate outputs for Qwen model
            qwencoder_generated_ids = qwencoder.generate(
                **qwencoder_inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )
            
            qwencoder_output = qwencoder_tokenizer.batch_decode(qwencoder_generated_ids)[0]
            output = qwencoder_output.replace('<|im_end|>', "").split("<|im_start|>assistant")[-1]

        elif model == "QwenCoderMedium":
            qwencoder_prompt = qwencodermedium_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )

            qwencoder_inputs = qwencodermedium_tokenizer([qwencoder_prompt], return_tensors="pt")
            # Generate outputs for Qwen model
            qwencoder_generated_ids = qwencodermedium.generate(**qwencoder_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
            qwencoder_output = qwencodermedium_tokenizer.batch_decode(qwencoder_generated_ids)[0]
            output = qwencoder_output.replace('<|im_end|>', "").split("<|im_start|>assistant")[-1]

        elif model == "QwenCoderSmall":
            qwencoder_prompt = qwencodersmall_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )

            qwencoder_inputs = qwencodersmall_tokenizer([qwencoder_prompt], return_tensors="pt")
            # Generate outputs for Qwen model
            qwencoder_generated_ids = qwencodersmall.generate(
                **qwencoder_inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )
            
            qwencoder_output = qwencodersmall_tokenizer.batch_decode(qwencoder_generated_ids)[0]
            output = qwencoder_output.replace('<|im_end|>', "").split("<|im_start|>assistant")[-1]

        elif model == "Codestral":
            mistralai_prompt = mistralai_prompt.replace('[[INPUT]]', prompt)
            mistral_inputs = codestral_tokenizer([mistralai_prompt], return_tensors="pt")
            mistral_generated_ids = codestral.generate(
                **mistral_inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                do_sample=True
            )
            output = codestral_tokenizer.batch_decode(mistral_generated_ids)[0]
            output = output.split("[/INST]")[-1].replace("</s>", "")

        
        updated_dataset_df["Prompt"][idx] = prompt
        updated_dataset_df["Model output"][idx] = output

    updated_dataset_df.to_csv(output_filename, sep=";", columns=list(updated_dataset_df.keys()), index=False)