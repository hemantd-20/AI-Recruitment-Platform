import pdfplumber
import docx2txt

def extract_text_from_pdf(pdf_file_object):
    """Extract text from a PDF resume."""
    text = ""
    with pdfplumber.open(pdf_file_object) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file_object):
    """Extract text from a DOCX resume."""
    return docx2txt.process(docx_file_object)

def read_text_from_file(file_path: str) -> str:
    """Read text from a file."""
    with open(file_path, "r") as file:
        return file.read()

