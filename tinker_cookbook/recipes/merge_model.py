from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "openai/gpt-oss-20b"
ADAPTER = ""  # folder containing adapter_model.safetensors
OUT = ""

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype="auto",
    device_map="cpu",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUT)
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.save_pretrained(OUT)

print(f"Done. Merged model saved to {OUT}")

