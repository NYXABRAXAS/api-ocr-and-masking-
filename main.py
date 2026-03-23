from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from starlette.background import BackgroundTasks

import pytesseract
from PIL import Image
import cv2
import re
import os
import json
import tempfile

app = FastAPI(title="Aadhaar OCR & Masking API")

# ---------------- API KEY ----------------
API_KEYS = ["mysecretkey123"]

def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ---------------- CLEANUP ----------------
def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# ---------------- HELPER ----------------
def clean_name(value):
    if not value:
        return value
    value = re.sub(r'[^A-Za-z\s]', '', value)
    words = [w for w in value.split() if len(w) > 1]
    return " ".join(words)

# ---------------- OCR FUNCTION ----------------
def run_ocr_with_boxes(image_path):
    img = cv2.imread(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data['text'])

    for i in range(n):
        text = data['text'][i].strip()
        if text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            words.append({
                "text": text,
                "bbox": (x, y, x + w, y + h)
            })

    full_text = " ".join([w["text"] for w in words])

    return img, words, full_text

# ---------------- MAIN API ----------------
@app.post("/v1/ocr/extract-and-mask")
async def extract_and_mask(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key)
):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only image files allowed")

    # Save temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    try:
        # ---------------- OCR ----------------
        img, words, full_text = run_ocr_with_boxes(input_path)

        lines = full_text.split()

        # ---------------- EXTRACTION ----------------
        extracted = {
            "aadhaar_number": None,
            "dob": None,
            "name": None
        }

        # Aadhaar Number
        num_match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', full_text)
        if num_match:
            extracted["aadhaar_number"] = num_match.group(0)

        # DOB
        dob_match = re.search(r'\d{2}/\d{2}/\d{4}', full_text)
        if dob_match:
            extracted["dob"] = dob_match.group(0)

        # Name detection (improved)
        possible_names = []
        for word in words:
            text = word["text"]
            if len(text.split()) >= 2 and text.isalpha():
                possible_names.append(text)

        if possible_names:
            extracted["name"] = clean_name(possible_names[0])

        # ---------------- MASKING ----------------
        if extracted["aadhaar_number"]:
            for word in words:
                text = word["text"].replace(" ", "")

                # Match Aadhaar parts
                if re.fullmatch(r'\d{4}', text) or re.fullmatch(r'\d{12}', text):
                    x1, y1, x2, y2 = word["bbox"]

                    # Mask first 2 blocks (approx)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Save masked image
        output_path = input_path.replace(suffix, f"_masked{suffix}")
        cv2.imwrite(output_path, img)

        # ---------------- RESPONSE ----------------
        headers = {
            "X-OCR-Data": json.dumps(extracted),
            "Access-Control-Expose-Headers": "X-OCR-Data"
        }

        background_tasks.add_task(remove_file, input_path)
        background_tasks.add_task(remove_file, output_path)

        return FileResponse(
            path=output_path,
            media_type=file.content_type,
            filename=f"masked_{file.filename}",
            headers=headers
        )

    except Exception as e:
        remove_file(input_path)
        raise HTTPException(500, str(e))


@app.get("/")
def home():
    return {"status": "OCR & Masking API Running 🚀"}