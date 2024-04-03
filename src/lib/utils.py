import torch
import transformers
from transformers import AutoModelForCausalLM


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
