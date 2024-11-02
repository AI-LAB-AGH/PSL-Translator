import transformers
import os
import definitions


class Translator:
    def __init__(self, model="translation/models/pretrained_models/allegro/plt5-large/checkpoint",
                 tokenizer="allegro/plt5-large"):
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            os.path.join(definitions.ROOT_DIR,
                         model)).to("cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    def translate(self, sentence):
        sentence = " ".join(sentence)
        outputs = self.model.generate(self.tokenizer(sentence, return_tensors="pt")['input_ids'].to("cpu"))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
