from openai import OpenAI
from translation.data_collection.gpt_api_key import get_api_key
from translation.data_collection.pdf_text_extractor import PdfTextExtractor


class GptDataCollector:
    """
    A class which manages text data collection using GPT.
    """

    def __init__(self, model: str, pdf_path: str):
        self.client = OpenAI(api_key=get_api_key())
        self.model = model
        self.pdf_text = PdfTextExtractor.extract_text_from_pdf(pdf_path)

    def prompt_gpt(self) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "Jesteś moim asystentem, który pomaga mi w zdobyciu danych do trenowania moich modeli uczenia maszynowego. Na postawie tekstu,"
                            "który podam ci na końcu wiadomości, musisz tworzyć poprawne zdania w polskim języku migowym wraz z tłumaczeniami. Generuj jedno zdanie na linię. "
                            "W twojej odpowiedzi mają znajdować się tylko zdania wraz z tłumaczeniami. Nie zawieraj w swojej odpowiedzi kropek na końcu zdań, numeracji ani pustych linii."
                            "Oto przykładowa para prompt - odpowiedź:"
                            "prompt: \"Wygeneruj mi 3 pary zdań\""
                            "odpowiedź: "
                            "\"Ty szkoła kiedy?\n"
                            "Kiedy masz szkołę?\n"
                            "Ja brat siostra mieć\n"
                            "Ja mam brata i siostrę\n"
                            "On gazeta czytać już\n"
                            "On przeczytał gazetę\n\""
                            "Staraj się układać różnorodne zdania, ale "
                            "stosuj się do zasad gramatyki języka migowego, które znajdziesz"
                            f"w następującym tekście: \"{self.pdf_text}\""},
                {"role": "user", "content": f"Wygeneruj mi 100 par zdań."}]
        )
        return completion.choices[0].message.content


def main():
    gpt = GptDataCollector("gpt-4o", "pjm_grammar.pdf")
    answer = gpt.prompt_gpt()
    with open("data.txt", "a", encoding="UTF-8") as f:
        f.write(answer)



if __name__ == "__main__":
    main()
