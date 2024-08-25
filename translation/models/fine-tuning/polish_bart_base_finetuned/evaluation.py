import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def bleu_evaluation(model, tokenizer, test_iter):
    score = 0
    for sample, target in test_iter:
        inputs = tokenizer.encode(sample, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
        target = target.split()
        if len(sample) < 3 or len(target) < 3:
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], prediction, weights=[0.5, 0.5])
        else:
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], prediction)
        score += BLEUscore
    return score / len(test_iter)

model = AutoModelForSeq2SeqLM.from_pretrained("./model_bart", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./model_bart", return_tensors="pt")