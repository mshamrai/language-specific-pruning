import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer
from importlib.metadata import version

from lib.prune import prune_wanda, prune_sparsegpt, check_sparsity
from lib.eval import eval_ppl_urbantext
from lib.data import get_ubertext, get_train_c4_test_ubertext
from lib.utils import get_memory_footprint, get_llm


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--load_bit", type=str, default=None)
    # parser.add_argument("--train_data_path", type=str, default=None)
    # parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--replace_sparse_layers", action="store_true", help="whether to use replace layers with 2:4 semi-sparse analogue")
    parser.add_argument("--eval_only", action="store_true", help="whether to only evaluate the dense model without pruning")
    parser.add_argument("--prune_on_c4", action="store_true", help="whether to prune the model on c4 dataset")

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
    if args.prune_on_c4:
        train_dataloader, test_dataloader = get_train_c4_test_ubertext(nsamples=args.nsamples,
                                                                        seed=args.seed,
                                                                        seqlen=model.seqlen,
                                                                        tokenizer=tokenizer)
    else:
        train_dataloader, test_dataloader = get_ubertext(nsamples=args.nsamples,
                                                        seed=args.seed,
                                                        seqlen=model.seqlen,
                                                        tokenizer=tokenizer)
    print("dataset loading complete")

    # ppl_test = eval_ppl_urbantext(model, test_dataloader, device)
    # print(f"perplexity before pruning: {ppl_test}")

    if not args.eval_only:

        if args.sparsity_ratio != 0:
            print("pruning starts")
            if args.prune_method == "wanda":
                prune_wanda(args, model, train_dataloader, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, train_dataloader, device, prune_n=prune_n, prune_m=prune_m)

        print("Memory footprint after pruning: " + get_memory_footprint(model))
        # memory_reporter = MemReporter(model)
        # memory_reporter.report()

        ################################################################
        print("*"*30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
        ################################################################
    ppl_test = eval_ppl_urbantext(model, test_dataloader, device)
    print(f"perplexity after pruning: {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{args.sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()