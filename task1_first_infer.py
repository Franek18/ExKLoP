import os
import torch
import argparse
import pandas as pd
import transformers

from rules_generator import get_python_rules

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default="Lllama-70", type=str, help="Which model to inference")
# parser.add_argument("--index", default=0, type=int, help="At which index start the dataset inference")
# args = parser.parse_args()


def task1_first_infer(model, index=0):

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


    dataset_df = pd.read_csv("data/dataset_premises.csv", delimiter=";", header=0)
    results = {"Prompt": [], model: []}
    output_filename = f"outputs/Task1/{model}_first_outputs.csv"

    system_prompt_filename = "templates/prompts/task1_first_system_prompt.txt"

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
        
    if not os.path.exists("outputs/Task1"):
        os.mkdir("outputs/Task1")

    # A variable for current number of parameters in an input prompt
    start_idx = index
    curr_no_params = 0

    for idx in dataset_df.index[start_idx:]:
        # A MistralAI template for a prompt
        f = open(system_prompt_filename)
        system_prompt = f.read()
        f.close()
        mistralai_prompt = f"[INST] {system_prompt}\n[[INPUT]]\n[/INST]"

        no_of_params = dataset_df["Number of parameters"][idx]

        # If we start next (greater) number of params than change In-context examples
        if curr_no_params < no_of_params:
            # Retrieve In-context examples
            example_text, example = get_python_rules(no_of_params)

            # A system prompt template for Llama3 and Qwen2
            # system_prompt = """System Message:
            # Generate a set of valid Python3 functions whic corresponds to the given set of logical rules expressed in natural language.
            # During the translation, please pay close attention to defining variables and rules.
            # Do not add any comments from you.
            # These are the rules:
            # """

            # A system prompt template for Llama3 and Qwen2
            # f = open(system_prompt_filename)
            # system_prompt = f.read()
            # f.close()

            # Modify input prompt
        system_prompt = system_prompt.replace('[[EXAMPLE_TEXT]]', example_text).replace('[[EXAMPLE]]', example)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
        ]

        curr_no_params = no_of_params
        
        mistralai_prompt = mistralai_prompt.replace('[[EXAMPLE_TEXT]]', example_text).replace('[[EXAMPLE]]', example)
        
        # Get a prompt from a dataset
        prompt = dataset_df["Prompt"][idx]

        # Prepare an input for Llama and Qwen models
        messages[1]["content"] = prompt

        full_prompt = system_prompt + "\n" + prompt

        messages[0]["content"] = ""
        messages[1]["content"] = full_prompt

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
                temperature=0,
                do_sample=False
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
                temperature=0,
                do_sample=False
            )


            output = llama_70b_outputs[0]["generated_text"].split("<|end_header_id|>")[-1]

        elif model == "Gemma3":
            # print(gemma_prompt + "\n")
            messages = [
                {
                    "role": "system",
                    # "content": [{"type": "text", "text": system_prompt}]
                    "content": [{"type": "text", "text": ""}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt}
                    ]
                }
            ]

            output = gemma_12b_pipeline(text_inputs=messages, max_new_tokens=2048, do_sample=False)
            output = output[0]["generated_text"][-1]["content"]

        elif model == "QwenCoder":
            qwencoder_prompt = qwencoder_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )

            qwencoder_inputs = qwencoder_tokenizer([qwencoder_prompt], return_tensors="pt")
            # Generate outputs for Qwen model
            qwencoder_generated_ids = qwencoder.generate(**qwencoder_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
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
            qwencoder_generated_ids = qwencodersmall.generate(**qwencoder_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
            qwencoder_output = qwencodersmall_tokenizer.batch_decode(qwencoder_generated_ids)[0]
            output = qwencoder_output.replace('<|im_end|>', "").split("<|im_start|>assistant")[-1]

        elif model == "Codestral":
            mistralai_prompt = mistralai_prompt.replace('[[INPUT]]', prompt)
            mistral_inputs = codestral_tokenizer([mistralai_prompt], return_tensors="pt")
            mistral_generated_ids = codestral.generate(**mistral_inputs, max_new_tokens=2048, temperature=0, do_sample=False)
            output = codestral_tokenizer.batch_decode(mistral_generated_ids)[0]
            output = output.split("[/INST]")[-1].replace("</s>", "")

        # print(output)
        # break

        results["Prompt"].append(full_prompt)
        results[model].append(output)

        if idx % 50 == 0:
            print(f"Idx == {idx} - saving batch of data")
            result_pd = pd.DataFrame(results)
            if not os.path.exists(output_filename):
                result_pd.to_csv(output_filename, sep=";", columns=list(results.keys()), index=False)
            else:
                result_pd.to_csv(output_filename, sep=";", mode='a', index=False, header=False)

            results["Prompt"] = []
            results[model] = []

    result_pd = pd.DataFrame(results)
    result_pd.to_csv(output_filename, sep=";", mode='a', index=False, header=False)