from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def load_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok
def generate_response(model, tok, question: str):
    inputs = tok([question], return_tensors="pt")

    streamer = TextStreamer(tok)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=128)
    return  # Optionally return the generated response if needed

# Load the model and tokenizer once
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model  , tok = load_model_and_tokenizer(model_name)
generate_response(model,tok,"An increasing sequence: one,")# Transformers Demo