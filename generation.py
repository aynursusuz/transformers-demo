from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed


def load_model_and_tokenizer(model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str):
    set_seed(42)

    messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt}
]
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    input_length = model_inputs.shape[1]
    generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=128, eos_token_id=model.config.eos_token_id)
    print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])

    return

prompt ="what is physics?"
system_message = "You are a helpful assistant."
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_name)
generate_response(model, tokenizer, prompt)

'''Physics is the branch of science that deals with the study of matter, energy, and their interactions. It encompasses many different sub-disciplines such as mechanics, electromagnetism, optics, thermodynamics, fluid dynamics, and more.

The main goals of physics are to understand how these phenomena work at an atomic or subatomic level, and then use this understanding to predict and control them on larger scales. Physics has applications in fields ranging from engineering and technology to medicine and astronomy.

Some key concepts in physics include:

1. Newton's laws: These are a set of three fundamental laws of motion that describe how objects move and interact with each '''
