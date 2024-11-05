import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import TER
from translation.translator import Translator


class Evaluator:
    def __init__(self, model, tokenizer, torch_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.translator = Translator(model, tokenizer)
        self.data = torch_dataset

    def evaluate(self):
        bleu = self.bleu_evaluation()
        print(f"evaluated {self.translator.model.name_or_path} with BLEU score: {bleu}")
        rouge = self.rouge_evaluation()
        print(f"evaluated {self.translator.model.name_or_path} with ROUGE score: {rouge}")
        meteor = self.meteor_evaluation()
        print(f"evaluated {self.translator.model.name_or_path} with METEOR score: {meteor}")
        ter = self.ter_evaluation()
        print(f"evaluated {self.translator.model.name_or_path} with TER score: {ter}")
        return {"bleu": bleu, "rouge": rouge, "meteor": meteor, "ter": ter, "model_name": self.translator.model.name_or_path}

    def bleu_evaluation(self):
        score = 0
        for sample, target in self.data:
            prediction = self.translator.translate(sample).split()
            target = target.split()
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], prediction,
                                                                weights= [0.5, 0.5])
            score += BLEUscore
        return score / len(self.data)

    def rouge_evaluation(self):
        rouge = Rouge()
        score = 0
        for sample, target in self.data:
            prediction = " ".join(self.translator.translate(sample).split())
            scores = rouge.get_scores(prediction, target)
            rouge_score = scores[0]['rouge-l']['f']  # F1 score for ROUGE-L
            score += rouge_score
        return score / len(self.data)

    def meteor_evaluation(self):
        score = 0
        for sample, target in self.data:
            prediction = self.translator.translate(sample).split()
            target = target.split()
            meteor = meteor_score([target], prediction)
            score += meteor
        return score / len(self.data)

    def ter_evaluation(self):
        ter = TER()
        score = 0
        for sample, target in self.data:
            prediction = " ".join(self.translator.translate(sample).split())
            ter_score = ter.sentence_score(prediction, [target]).score
            score += ter_score
        return score / len(self.data)
