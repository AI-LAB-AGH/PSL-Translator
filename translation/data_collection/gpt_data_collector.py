from openai import OpenAI
from translation.data_collection.gpt_api_key import get_api_key
from translation.data_collection.pdf_text_extractor import PdfTextExtractor

pdf = PdfTextExtractor.extract_text_from_pdf("pjm_grammar.pdf")

incorrect = "Jesteś moim asystentem, który pomaga mi w zdobyciu danych do trenowania moich modeli uczenia maszynowego. Chciałbym żebyś tworzył niepoprawne, ucięte zdania,"
"w szczególności bez żadnego czasownika. Generuj jedno zdanie na linię. Proszę, aby po każdym wygenerowanym zdaniu, w kolejnej linii znalazło się identyczne. "
"W twojej odpowiedzi mają znajdować się tylko zdania wraz z duplikatami. Nie zawieraj w swojej odpowiedzi kropek na końcu zdań, numeracji ani pustych linii."
"Oto przykładowa para prompt - odpowiedź:"
"prompt: \"Wygeneruj mi 5 par zdań\""
"odpowiedź: "
"\"Ja książka w poniedziałek\n"
"Ja książka w poniedziałek\n"
"On gazeta\n"
"On gazeta\n"
"Ty szkoła\n"
"Ty szkoła\n"
"Ja brat siostra\n"
"Ja brat siostra\n"
"On rower nowy.\n"
"On rower nowy.\n"

correct = "Jesteś moim asystentem, który pomaga mi w zdobyciu danych do trenowania moich modeli uczenia maszynowego. Na postawie tekstu,"
"który podam ci na końcu wiadomości, musisz tworzyć poprawne zdania w polskim języku migowym wraz z tłumaczeniami. "
"Generuj jedno zdanie na linię. W twojej odpowiedzi mają znajdować się tylko zdania wraz z tłumaczeniami."
"Oto przykładowa para prompt - odpowiedź:"
"prompt: \"Wygeneruj mi 6 par zdań\""
"odpowiedź: "
"\"Ja książka w poniedziałek czytać było\n"
"Czytałem książkę w poniedziałek\n"
"Ty mieszkać gdzie?\n"
"Gdzie mieszkasz?\n"
"On gazeta czytać już\n"
"On przeczytał gazetę\n"
"Ty szkoła kiedy?\n"
"Kiedy masz szkołę?\n"
"Ja brat siostra mieć\n"
"Ja mam brata i siostrę\n"
"On rower nowy kupić.\n"
"On kupił nowy rower\n\""
"Staraj się układać różnorodne zdania, ale "
"stosuj się do zasad gramatyki języka migowego. "
"W szczególności stosuj się do poprawnego szyku zdania. Zasady gramatyki "
"polskiego języka migowego znajdziesz"
f"w następującym tekście: \"{pdf}\""


class GptDataCollector:
    """
    A class which manages text nlp_data collection using GPT.
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
"który podam ci na końcu wiadomości, musisz tworzyć poprawne zdania w polskim języku migowym wraz z tłumaczeniami. "
"Generuj jedno zdanie na linię. W twojej odpowiedzi mają znajdować się tylko zdania wraz z tłumaczeniami."
"Oto przykładowa para prompt - odpowiedź:"
"prompt: \"Wygeneruj mi 6 par zdań\""
"odpowiedź: "
"\"Ja książka w poniedziałek czytać było\n"
"Czytałem książkę w poniedziałek\n"
"Ty mieszkać gdzie?\n"
"Gdzie mieszkasz?\n"
"On gazeta czytać już\n"
"On przeczytał gazetę\n"
"Ty szkoła kiedy?\n"
"Kiedy masz szkołę?\n"
"Ja brat siostra mieć\n"
"Ja mam brata i siostrę\n"
"On rower nowy kupić.\n"
"On kupił nowy rower\n\""
"Staraj się układać różnorodne zdania, ale "
"stosuj się do zasad gramatyki języka migowego. "
"W szczególności stosuj się do poprawnego szyku zdania. Zasady gramatyki "
"polskiego języka migowego znajdziesz"
f"w następującym tekście: \"{self.pdf_text}\""},
                {"role": "user", "content": f"Wygeneruj mi 100 par zdań."}]
        )
        return completion.choices[0].message.content


def main():
    for i in range(100):
        gpt = GptDataCollector("gpt-4o", "pjm_grammar.pdf")
        answer = gpt.prompt_gpt()
        with open("../nlp_data/more_data_backup.txt", "a+", encoding="UTF-8") as f:
            f.write("\n")
            f.write(answer)



if __name__ == "__main__":
    main()