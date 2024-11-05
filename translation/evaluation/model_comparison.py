import transformers
import os
import definitions
from translation.evaluation.evaluator import Evaluator
from translation.torch_datasets.translation_dataset import TranslationDataset
import nltk
import pandas as pd
import torch
from translation.models.transformer.model import Seq2SeqTransformer
from translation.models.transformer.processing_helpers import translate as translate_transformer
nltk.download('wordnet')


def evaluate_hf_model(model_path, tokenizer_name, data_path):
    data = TranslationDataset(os.path.join(definitions.ROOT_DIR, data_path), prompt=False)
    evaluator = Evaluator(model_path, tokenizer_name, data)
    return evaluator.evaluate()


def evaluate_transformer(model_path, tokenizer_name, data_path):
    pass




def main():
    results = []
    results.append(evaluate_hf_model("translation/models/pretrained_models/allegro/plt5-large-more-data/checkpoint",
                                     "allegro/plt5-large", "translation/nlp_data/test_data.txt"))
    results.append(evaluate_hf_model("translation/models/pretrained_models/allegro/plt5-large-more-data-1/checkpoint",
                                     "allegro/plt5-large", "translation/nlp_data/test_data.txt"))
    results.append(evaluate_hf_model("translation/models/pretrained_models/allegro/plt5-large-more-data-2/checkpoint",
                                     "allegro/plt5-large", "translation/nlp_data/test_data.txt"))

    df = pd.DataFrame(results)
    df.set_index("model_name", inplace=True)
    df.to_csv("model_comparison_updated_test_set.csv")


if __name__ == "__main__":
    main()
