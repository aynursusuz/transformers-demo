from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto", load_in_4bit=False)

set_seed(42)
messages = [
    {
        "role": "system",
        "content": "I will now answer your questions as if I were a physics professor. ",
    },
    {"role": "user", "content": "Can you tell me , physics ?"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
input_length = model_inputs.shape[1]
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=128, eos_token_id=model.config.eos_token_id)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])

'''Certainly! Physics is the fundamental study of matter and energy, their interactions, and the laws governing them. It deals with concepts such as motion, force, mass, energy, space, time, and the relationships between these physical quantities.

Physics encompasses several key areas:
1. **Mechanics**: This branch focuses on the behavior of objects in motion or at rest.
2. **Elasticity**: Deals with the properties of materials that allow them to return to their original shape after deformation.
3. **Thermodynamics**: Studies the relationship between heat, work, and energy transformations in closed systems.
4. **Electromagnetism'''
