import PyPDF2
import re
import string


class PdfTextExtractor:
    """
    A class which manages pdf text extraction.
    """
    @classmethod
    def extract_text_from_pdf(cls, pdf_path: str) -> str:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

            text = cls.clean_string(text)
            return text

    @classmethod
    def clean_string(cls, text: str) -> str:
        pattern = f"[^{re.escape(string.ascii_letters + string.digits + string.punctuation)}\s]"
        return re.sub(pattern, "", text)
