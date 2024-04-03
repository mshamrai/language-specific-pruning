# prune using unstructured SparseGPT and UberText  
python main.py  --model baffo32/decapoda-research-llama-7B-hf \
                --prune_method sparsegpt \
                --sparsity_ratio 0.5 \
                --sparsity_type unstructured \
                --nsamples 64 \
                --seed 0

# prune using 2:4 SparseGPT with replacement to semi-structured linear layers
python main.py  --model baffo32/decapoda-research-llama-7B-hf \
                --prune_method sparsegpt \
                --sparsity_ratio 0.5 \
                --sparsity_type "2:4" \
                --replace_sparse_layers \
                --nsamples 64 \
                --seed 0

# eval dense model on UberText
python main.py  --model baffo32/decapoda-research-llama-7B-hf \
                --eval_only \
                --seed 0

# prune on c4 eval on UberText
python main.py  --model baffo32/decapoda-research-llama-7B-hf \
                --prune_method sparsegpt \
                --sparsity_ratio 0.5 \
                --sparsity_type "2:4" \
                --prune_on_c4 \
                --nsamples 64 \
                --seed 0
