from src.textSummariser.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
    
    def predict(self, text):
        # 1. Load the tokenizer and model directly
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        print("Dialogue:")
        print(text)

        # 2. Add the T5 prefix and tokenize (with truncation so it never crashes!)
        formatted_text = "summarize: " + text
        inputs = tokenizer(formatted_text, max_length=1024, truncation=True, return_tensors="pt")

        # 3. Generate the summary using the model
        summaries = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            length_penalty=0.8, 
            num_beams=2, 
            max_length=128
        )

        # 4. Decode the output back into readable text
        output = tokenizer.decode(summaries[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print("\nModel Summary:")
        print(output)

        return output