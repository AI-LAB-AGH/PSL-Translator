import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import TER


class Evaluator:
    def __init__(self, model, tokenizer, translate, torch_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.translate = translate
        self.data = torch_dataset

    def evaluate(self):
        bleu = self.bleu_evaluation()
        rouge = self.rouge_evaluation()
        meteor = self.meteor_evaluation()
        ter = self.ter_evaluation()
        return {"bleu": bleu, "rouge": rouge, "meteor": meteor, "ter": ter}

    def bleu_evaluation(self):
        score = 0
        for sample, target in self.data:
            prediction = self.translate(sample).split()
            print(sample, "-", " ".join(prediction))
            target = target.split()
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], prediction,
                                                                weights=[1 / len(sample) for i in range(len(sample))])
            score += BLEUscore
        return score / len(self.data)

    def rouge_evaluation(self):
        rouge = Rouge()
        score = 0
        for sample, target in self.data:
            prediction = " ".join(self.translate(sample).split())
            print(sample, "-", prediction)
            scores = rouge.get_scores(prediction, target)
            rouge_score = scores[0]['rouge-l']['f']  # F1 score for ROUGE-L
            score += rouge_score
        return score / len(self.data)

    def meteor_evaluation(self):
        score = 0
        for sample, target in self.data:
            prediction = " ".join(self.translate(sample).split())
            print(sample, "-", prediction)
            meteor = meteor_score([target], prediction)
            score += meteor
        return score / len(self.data)

    def ter_evaluation(self):
        ter = TER()
        score = 0
        for sample, target in self.data:
            prediction = " ".join(self.translate(sample).split())
            print(sample, "-", prediction)
            ter_score = ter.sentence_score(prediction, [target]).score
            score += ter_score
        return score / len(self.data)
