# Language-Specific Pruning for Efficient Reduction of Large Language Models

This repository contains the code implementation for the paper "Language-Specific Pruning for Efficient Reduction of Large Language Models".


## Overview

This project focuses on exproring pruning techniques tailored to specific languages to enhance the efficiency of Large Language Models (LLMs). The key contribution lies in recognizing distinct language-specific weight distributions in LLMs trained on diverse languages. By leveraging this insight, the pruning process effectively compresses LLMs while preserving competitive performance. The code provided here allows for the reproduction of the results presented in the associated paper.


## Usage

To prune a model using unstructured SparseGPT and UberText:

```bash
python main.py  --model baffo32/decapoda-research-llama-7B-hf \
                --prune_method sparsegpt \
                --sparsity_ratio 0.5 \
                --sparsity_type unstructured \
                --nsamples 64 \
                --seed 0
```

The arguments:  
- `--model`: The identifier for the model from Hugging Face model hub.
- `--seed`: Seed for sampling the calibration data.
- `--nsamples`: Number of calibration samples.
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--prune_method`: Pruning method to use, namely [`wanda`, `sparsegpt`].
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.
- `--save_model`: Path to save the pruned model.
- `--load_bit`: Whether to use load quantized model (mostly for evaluation).
- `--replace_sparse_layers`: Whether to replace linear layers with 2:4 semi-sparse analogue.
- `--eval_only`: Whether to only evaluate the dense model without pruning.
- `--prune_on_c4`: Whether to prune the model on c4 dataset.
- `--device`: Device where to load the model (default is `cuda:0`).

For more examples of usage see [launch.sh](src/launch.sh)


## Dataset

The Ukrainian dataset for calibration sampled from UberText 2.0 and published to HF: [mshamrai/lang-pruning-uk-uber-text-2](https://huggingface.co/datasets/mshamrai/lang-pruning-uk-uber-text-2)


## Acknowledgement

The main codebase was adapted from the [wanda](https://github.com/locuslab/wanda) repository.


## License

This project is licensed under the MIT License. Feel free to use and modify the code for academic and research purposes.

For inquiries, please contact m.shamrai at imath.kiev.ua 