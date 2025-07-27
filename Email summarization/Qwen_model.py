from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch



class Qwen_model():
    def __init__(self): 
        # Set Hugging Face cache directory
        os.environ["HF_HOME"] = "D:/HuggingFace"

        # Define cache directory
        model_name="Qwen/Qwen2.5-0.5B-Instruct"
        cache_dir = "D:/HuggingFace/hub"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16,)

        self.prompt =[]



    def set_system_role(self, system_message="You are a helpful assistant."):
        self.system_role = {"role": "system", "content": system_message}
        self.prompt.append(self.system_role)

    def clear_prompt(self):  
        self.prompt.append(self.system_role)  

    def run_query(self,query):
        if query:
            self.prompt.append({"role":"user", "content": query})

            # applaying a template for cahting with the language model
            text = self.tokenizer.apply_chat_template(self.prompt,tokenize=False,add_generation_prompt=True)

            #return the tokens in pytorch tensor not python list and use the same device as the model (CPU in my case)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device) 

            #generate the output with max tokens of 512
            generated_ids = self.model.generate(**model_inputs,max_new_tokens=512)

            #remove the input tokens from the output tokens 
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.clear_prompt()
            return response

