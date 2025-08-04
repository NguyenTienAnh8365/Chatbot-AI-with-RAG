import torch
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from huggingface_hub import login

# Đăng nhập với token Hugging Face
login(token="your api key here")

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

def get_hf_llm(model_name: str="meta-llama/Llama-3.2-3B-Instruct", 
               max_new_tokens=1024, 
               **kwargs):
    # Tải cấu hình và ghi đè rope_scaling
    config = AutoConfig.from_pretrained(model_name)
    # config.rope_scaling = {"type": "llama3", "factor": 32.0}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        quantization_config=nf4_config, 
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )

    return llm