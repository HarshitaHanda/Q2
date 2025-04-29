import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

TESSERACT_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

KNOWN_UNITS = [
    r"%", r"g/dl", r"mg/dl", r"million/cu\.mm", r"cells/cu\.mm",
    r"fl", r"pg", r"sec(?:onds)?", r"iu/l", r"µg/dl",
]


class LabTest(BaseModel):
    test_name: str
    obtained_value: str
    bio_reference_range: str
    unit: Optional[str] = None
    lab_test_out_of_range: Optional[bool] = None


class LabResponse(BaseModel):
    is_success: bool
    data: dict


def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)


def run_ocr(img: np.ndarray) -> List[str]:
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    lines = [re.sub(r"\s{2,}", " ", ln.strip()) for ln in text.splitlines()]
    return [ln for ln in lines if ln]


RANGE_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)")
VALUE_PAT = re.compile(r"[-↑↓]?\s*(\d+(?:\.\d+)?)")
UNIT_PAT = re.compile("|".join(KNOWN_UNITS), flags=re.IGNORECASE)


def parse_line(line: str) -> Optional[LabTest]:
    range_match = RANGE_PAT.search(line)
    if not range_match:
        return None
    range_txt = range_match.group(0)

    tmp = line.replace(range_txt, " ")
    value_match = VALUE_PAT.search(tmp)
    if not value_match:
        return None
    value_txt = value_match.group(1)

    unit_match = UNIT_PAT.search(tmp)
    unit_txt = unit_match.group(0) if unit_match else None

    value_pos = tmp.find(value_match.group(0))
    test_name = tmp[:value_pos].strip(" :-")
    if not test_name or len(test_name) < 2:
        return None

    try:
        lo, hi = map(float, range_match.groups())
        val = float(value_txt)
        out_of_range = val < lo or val > hi
    except Exception:
        out_of_range = None

    return LabTest(
        test_name=test_name.upper(),
        obtained_value=value_txt,
        bio_reference_range=range_txt,
        unit=unit_txt,
        lab_test_out_of_range=out_of_range,
    )


def extract_tests(lines: List[str]) -> List[LabTest]:
    tests: List[LabTest] = []
    for ln in lines:
        row = parse_line(ln)
        if row:
            tests.append(row)
    return tests


app = FastAPI(
    title="Lab-Report Extractor",
    description="Extracts lab test name, value, reference range & unit from a diagnostic report image",
    version="1.0.0",
)


@app.post("/get-lab-tests", response_model=LabResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "application/pdf"):
        raise HTTPException(status_code=400, detail="Only PNG, JPEG or single-page PDF accepted")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        with tmp_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        if tmp_path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                page = convert_from_path(str(tmp_path), dpi=300)[0]
                img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            except ImportError:
                raise HTTPException(status_code=500, detail="pdf2image not installed: `pip install pdf2image`")
        else:
            img = cv2.imdecode(np.fromfile(tmp_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image")

    processed = preprocess_image(img)
    lines = run_ocr(processed)
    lab_tests = extract_tests(lines)

    return JSONResponse(
        status_code=200,
        content=LabResponse(
            is_success=True,
            data={"lab_tests": [t.dict() for t in lab_tests]},
        ).dict(),
    )
