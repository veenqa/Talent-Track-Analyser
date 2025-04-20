!pip install pytesseract
!sudo apt install tesseract-ocr
!sudo apt install libtesseract-dev
!pip install python-docx
!pip install pdf2image
!sudo apt install tesseract-ocr
!pip install pymupdf
!apt-get install poppler-utils # Installs Poppler on Debian/Ubuntu-based systems
!pip install pdf2image        # Installs the pdf2image library
!pip install datasets


import os
import re
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
from transformers import BertForSequenceClassification, BertTokenizer
from torch import cuda

# Check for GPU
device = "cuda" if cuda.is_available() else "cpu"

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Placeholder for label encoder
job_categories = {
    0: "Software Engineer",
    1: "Advocate",
    2: "Data Scientist",
    3: "Project Manager"  # You can dynamically expand this
}

# Keywords for job categorization (sample keywords, can be expanded)
category_keywords = {
    "Software Engineer": [
        "python", "java", "c++", "software", "developer", "machine learning", "AI", "web development", "data science", "algorithm", "frontend", "backend", "react", "django", "flask", "SQL", "database", "cloud", "devops", "automation"
    ],
    "Advocate": [
        "court", "law", "lawyer", "litigation", "advocate", "legal", "attorney", "judiciary", "justice", "civil law", "criminal law", "legal advisor"
    ],
    "Data Scientist": [
        "data science", "machine learning", "AI", "data analysis", "python", "R", "big data", "data visualization", "deep learning", "neural networks", "statistical analysis", "SQL", "NLP"
    ],
    "Project Manager": [
        "project management", "scrum", "agile", "leadership", "team management", "timeline", "stakeholders", "planning", "sprint", "deliverables"
    ]
}

def preprocess_text(text):
    """
    Preprocesses the input text by removing special characters, converting to lowercase,
    and removing extra whitespace.
    """
    # Keep alphanumeric characters and spaces only
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(' +', ' ', text).strip()

    return text

def extract_text_from_file(file_path):
    """
    Extracts text from various file types (PDF, DOCX, image).
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        # Process image file
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)

    elif ext == '.pdf':
        # Process PDF file
        pages = convert_from_path(file_path)  # Convert PDF to images
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)  # Extract text from images
        return text

    elif ext == '.docx':
        # Process Word document
        doc = Document(file_path)
        text = " ".join([p.text for p in doc.paragraphs])
        return text

    else:
        raise ValueError("Unsupported file format. Please upload an image, PDF, or DOCX file.")

def classify_resume(text, model, tokenizer, label_encoder):
    """
    Classifies the resume based on the preprocessed text using the model and tokenizer.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax().item()
    return label_encoder[predicted_label]

def process_uploaded_file(file_path):
    """
    Processes the uploaded resume file, extracts text, classifies it, and displays the results.
    """
    # Extract text from the file
    extracted_text = extract_text_from_file(file_path)

    # Preprocess the extracted text
    clean_text = preprocess_text(extracted_text)

    # Classify the resume and get the category
    predicted_category = classify_resume(clean_text, model, tokenizer, job_categories)

    # Print the extracted text and the predicted category
    print(f"Extracted Resume Text:\n{extracted_text}\n")
    print(f"Predicted Resume Category: {predicted_category}")

# Example usage:
file_path = "/content/drive/MyDrive/resume.pdf"  # Replace with the actual file path
process_uploaded_file(file_path)
