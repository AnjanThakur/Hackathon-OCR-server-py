# app.py

from flask import Flask, request, jsonify
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import tempfile
import magic
import google.generativeai as genai
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

# ---------- Load environment variables and Configure Services ----------
load_dotenv()

# 1. Configure Tesseract executable path
# Ensure this path is correct for your system.
if platform.system() == "Windows":
    # Local development on your laptop
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
else:
    # On Railway (Linux), tesseract will be installed in PATH
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
# 2. Configure Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")
genai.configure(api_key=api_key)

# 3. Initialize the Gemini Model (best practice: do this once)
# Using 'gemini-1.5-flash-latest' ensures you're using the most recent version.
model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = Flask(__name__)

# ---------- OCR Function ----------
def ocr_pdf_or_image(file_path: str) -> List[Dict[str, Any]]:
    """
    Performs OCR on a given file (PDF or image) and returns text per page.
    """
    text_blocks = []
    file_type = magic.from_file(file_path, mime=True)
    is_pdf = file_path.lower().endswith('.pdf') or 'pdf' in file_type

    if is_pdf:
        # For PDFs, convert each page to an image and then OCR
        try:
            images = convert_from_path(file_path)
            for i, img in enumerate(images, start=1):
                page_text = pytesseract.image_to_string(img)
                text_blocks.append({"page": i, "text": page_text})
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF. Is Poppler installed and in your PATH? Error: {e}")
    else:
        # For images, directly perform OCR
        img = Image.open(file_path)
        page_text = pytesseract.image_to_string(img)
        text_blocks.append({"page": 1, "text": page_text})

    return text_blocks

# ---------- Gemini Extraction Function ----------
def extract_entities_with_gemini(text: str) -> Dict[str, Any]:
    """
    Uses Gemini to extract structured medical information from text.
    """
    prompt = f"""
    Extract structured medical information from the following OCR text.
    Return a valid JSON object with these fields:
    - "name": Patient's full name (string)
    - "age": Patient's age (integer or string)
    - "disease": Primary disease or condition diagnosed (string)
    - "values": A dictionary of lab/test values with their units (e.g., {{"Hemoglobin": "14 g/dL"}})
    - "doctorInfo": A dictionary with the doctor's details ("name", "clinic", "address", "phone")
    - "patientVitals": A dictionary of vitals ("weight", "bloodPressure", "temperature", "pulse", "height", "bmi")
    - "prescriptions": A list of dictionaries, where each contains {{"medication", "dosage", "timings", "duration"}}
    - "notes": Any additional clinical notes or advice (string)

    OCR Text:
    ---
    {text}
    ---

    Return valid JSON only. If a field is not present, use null or an empty structure (e.g., [], {{}}, "").
    """
    
    # Set the generation config to ensure the output is JSON
    generation_config = genai.GenerationConfig(response_mime_type="application/json")

    # The new syntax: call generate_content on the model instance
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )

    try:
        parsed_json = json.loads(response.text)
        return parsed_json
    except json.JSONDecodeError:
        print(f"Gemini response was not valid JSON. Raw text: {response.text}")
        return {"error": "Failed to parse Gemini response as JSON", "raw_text": response.text}
    except Exception as e:
        print(f"An unexpected error occurred during Gemini response handling: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}", "raw_text": response.text}

# ---------- API Endpoint ----------
@app.route('/extract-entities', methods=['POST'])
def extract_entities_endpoint():
    """
    Flask endpoint to upload a file, perform OCR, and extract entities.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_ext = os.path.splitext(file.filename)[1]
    
    # Use a temporary file to securely handle the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        file_path = tmp_file.name
        file.save(file_path)
    
    try:
        # Step 1: Perform OCR
        print(f"Performing OCR on temporary file: {file_path}")
        pages = ocr_pdf_or_image(file_path)
        full_text = "\n\n--- Page Break ---\n\n".join([p["text"] for p in pages])
        
        if not full_text.strip():
            return jsonify({'error': 'OCR could not extract any text from the document.'}), 400

        # Step 2: Extract entities using Gemini
        print("Sending OCR text to Gemini for entity extraction...")
        gemini_json = extract_entities_with_gemini(full_text)

        # Step 3: Build response
        response_data = {
            "filename": file.filename,
            "ocr_text": full_text,
            "extracted_entities": gemini_json
        }

        # ✅ Print final response in terminal
        print("Final JSON response:")
        print(json.dumps(response_data, indent=2, ensure_ascii=False))

        # Step 4: Return the successful response
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
    
    finally:
        # Step 5: Clean up the temporary file in all cases
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"Cleaned up temporary file: {file_path}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
