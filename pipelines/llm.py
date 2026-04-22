from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM_Pipe:
    def __init__(self, model_id, device='auto'):
        self.load_model(model_id, device)
    
    def load_model(self, model_id, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            dtype="bfloat16",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def inference(self, prompt, max_new_tokens=1024):
        inputs_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)
        inputs = inputs_ids['input_ids']
        output = self.model.generate(
            **inputs_ids,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
        )
        
        generated_tokens = output[0][inputs.shape[-1]:]
    
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)