from openai import OpenAI
from data.translation.data_generation import api_key

client = OpenAI(api_key=api_key.get_api_key())
MODEL = "gpt-4o"


words = []
with open("../words.txt", 'r', encoding='utf-8') as file:
  for line in file:
    word = line.strip()
    words.append(word)

words_string = ", ".join(words)

for i in range(300):
    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "Jesteś moim asystentem, który pomaga mi w zdobyciu danych do trenowania moich modeli uczenia maszynowego."},
        {"role": "user", "content": f"Chciałbym żebyś pomógł mi zdobyć dane do trenowania mojego modelu uczenia maszynowego."
                                    "Na końcu tej wiadomości podam ci po przecinku listę słów. Chcę żebyś ułożył 50 zdań bezpośrednio z tych słów, bez zmieniania ich formy. "
                                    "Chcę żeby każde zdanie miało co najmniej 6 słów."
                                    "Nie będą one poprawnie gramatycznie, ale upewnij sie, że mają sens semantyczny.  "
                                    "Do każdego zdania dopisz poprawioną jego wersję, czyli zmień formy słów, tak by zdanie było poprawnie gramatycznie. "
                                    "Proszę aby twoja wiadomość zawierała jedynie ułożone oraz poprawione zdania, każde oddzielone znakiem końca linii, "
                                    "bez numerów porządkowych. Nie rozdzielaj par zdań pustymi liniami ani słów przecinkami."
                                    "Na przykład:"
                                    "ona lubić jeść śniadanie swój mama. \n"
                                    "Ona lubi jeść śniadanie ze swoją mamą. \n"
                                    "ja Spać, ona zadzwonić. \n"
                                    "Ja spałem, gdy ona zadzwoniła. \n"
                                    "Oni mieszkać w małe mieszkanie lubić roślina w ogród   \n"
                                    "Oni mieszkają w małym mieszkaniu i lubią rośliny w ogrodzie. \n"
                                    "To koniec przykładów."
                                    "Oto lista słów z których masz tworzyć zdania: " + words_string}
      ]
    )
    answer = completion.choices[0].message.content

    with open("third_data.txt", "a+") as f:
        f.write(answer)
        f.write("\n")
