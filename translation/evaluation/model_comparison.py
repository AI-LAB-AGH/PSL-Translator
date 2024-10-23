import transformers
import os
import definitions
from translation.evaluation.evaluator import Evaluator
from translation.torch_datasets.translation_dataset import TranslationDataset


def evaluate_hf_model(model_path, tokenizer_name, data_path):
    def translate(sentence):
        outputs = model.generate(tokenizer(sentence, return_tensors="pt")['input_ids'].to("cuda"))
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        os.path.join(definitions.ROOT_DIR, model_path)).to('cuda')
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    data = TranslationDataset(os.path.join(definitions.ROOT_DIR, data_path))

    evaluator = Evaluator(model, tokenizer, translate, data)

    return evaluator.evaluate()


def evaluate_torch_model(path):
    pass


def main():
    print(evaluate_hf_model("translation/models/pretrained_models/mt5-base/checkpoint",
                            "google/mt5-base", "translation/nlp_data/test_data.txt"))


if __name__ == "__main__":
    main()
