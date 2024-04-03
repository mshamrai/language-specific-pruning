import argparse
import os 
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot, eval_ppl_urbantext
from lib.data import get_urbantext, get_c4


def get_memory_footprint(model):
    print(set([type(param.data) for param in model.parameters()]))
    mem_params = 0
    for param in model.parameters():
        t = param.data
        if isinstance(t, torch.sparse.semi_structured.SparseSemiStructuredTensor):
            mem_params += t.indices().nelement() * t.indices().element_size() + t.values().nelement() * t.values().element_size()
        else:
            mem_params += param.nelement()*param.element_size()

    # mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    mem = mem / 2**30 # in Gbs
    mem_str = f"{mem:.2f} Gbs"
    return mem_str


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(load_bit, model_name, cache_dir="llm_weights"):
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

    print(model.config.max_position_embeddings)
    model.seqlen = min(model.config.max_position_embeddings, 4096)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--load_bit", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.load_bit, args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    # model = model.to(device)
    print("Memory footprint before pruning: " + get_memory_footprint(model))
    # memory_reporter = MemReporter(model)
    # memory_reporter.report()
    # del memory_reporter
    
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

    train_dataloader, _ = get_c4(nsamples=args.nsamples,
                                                      seed=args.seed,
                                                      seqlen=model.seqlen,
                                                      tokenizer=tokenizer,)
    print("dataset loading complete")

    # ppl_test = eval_ppl_urbantext(model, test_dataloader, device)
    # print(f"perplexity before pruning: {ppl_test}")

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, train_dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, train_dataloader, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    print("Memory footprint after pruning: " + get_memory_footprint(model))
    # memory_reporter = MemReporter(model)
    # memory_reporter.report()

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    # ppl_test = eval_ppl(args, model, tokenizer, device)
    ppl_test = eval_ppl_urbantext(model, test_dataloader, device)
    print(f"perplexity after pruning: {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{args.sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()