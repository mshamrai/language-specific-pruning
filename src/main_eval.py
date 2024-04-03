import argparse
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.eval import eval_ppl_urbantext
from lib.data import get_urbantext 
from peft import PeftModel


def get_llm(load_bit, model_name, lora, cache_dir="llm_weights"):
    if load_bit == "4":
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            # torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            # low_cpu_mem_usage=True, 
            # device_map="auto",
            quantization_config=bnb_config
        )
    elif load_bit == "8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir, 
            device_map="auto",
            load_in_8bit=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            # low_cpu_mem_usage=True, 
            device_map="auto",
        )

    if lora:
        model = PeftModel.from_pretrained(model, lora)
    print(model.config.max_position_embeddings)
    model.seqlen = min(model.config.max_position_embeddings, 4096)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--lora', type=str, default=None, help='Lora to the LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')

    parser.add_argument("--load_bit", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.load_bit, args.model, args.lora, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    print("loading calibdation data")
    _, test_dataloader = get_urbantext(nsamples=args.nsamples,
                                                      seed=args.seed,
                                                      seqlen=model.seqlen,
                                                      tokenizer=tokenizer,
                                                      train_data_path=args.train_data_path, 
                                                      test_data_path=args.test_data_path)
    print("dataset loading complete")

    ppl_test = eval_ppl_urbantext(model, test_dataloader, device)
    print(f"perplexity before pruning: {ppl_test}")


if __name__ == '__main__':
    main()