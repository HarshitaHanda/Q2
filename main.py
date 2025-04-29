"""
FastAPI service to extract lab-test rows from a diagnostic report image.

Author: (your-name)
Date  : 29-Apr-2025
"""

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

# ----------------------------------------------------------------------
# -------------  CONFIGURATION  ----------------------------------------
# ----------------------------------------------------------------------
TESSERACT_CONFIG = (
    "--oem 3 "
    "--psm 6 "          # assume a single uniform block of text
    "-c preserve_interword_spaces=1"
)
# Common units seen in reports (extend as required)
KNOWN_UNITS = [
    r"%", r"g/dl", r"mg/dl", r"million/cu\.mm", r"cells/cu\.mm",
    r"fl", r"pg", r"sec(?:onds)?", r"iu/l", r"µg/dl",
]

# ----------------------------------------------------------------------
# -------------  DATA MODELS  ------------------------------------------
# ----------------------------------------------------------------------
class LabTest(BaseModel):
    test_name: str
    obtained_value: str
    bio_reference_range: str
    unit: Optional[str] = None
    lab_test_out_of_range: Optional[bool] = None


class LabResponse(BaseModel):
    is_success: bool
    data: dict


# ----------------------------------------------------------------------
# -------------  IMAGE PRE-PROCESSING  ---------------------------------
# ----------------------------------------------------------------------
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Light denoise + threshold to help OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove light shadows / noise
    blur = cv2.medianBlur(gray, 3)
    # adaptive threshold keeps tables/grid lines readable
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=15
    )
    return thresh


# ----------------------------------------------------------------------
# -------------  OCR UTILITIES  ----------------------------------------
# ----------------------------------------------------------------------
def run_ocr(img: np.ndarray) -> List[str]:
    """
    Returns a list of text lines (left-to-right order) from an image.
    """
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    # split → normalise whitespace
    lines = [re.sub(r"\s{2,}", " ", ln.strip()) for ln in text.splitlines()]
    return [ln for ln in lines if ln]   # drop blanks


# ----------------------------------------------------------------------
# -------------  ROW PARSER  -------------------------------------------
# ----------------------------------------------------------------------
RANGE_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)")
VALUE_PAT = re.compile(r"[-↑↓]?\s*(\d+(?:\.\d+)?)")
UNIT_PAT  = re.compile("|".join(KNOWN_UNITS), flags=re.IGNORECASE)


def parse_line(line: str) -> Optional[LabTest]:
    """
    Heuristic parser for a single OCR line.  Works well for most
    tabular reports where a row contains:
        <name> <value> [unit] <range>

    Returns LabTest or None if the line does not match.
    """
    # 1. reference range ------------------------------------------------
    range_match = RANGE_PAT.search(line)
    if not range_match:
        return None  # no range → skip (likely a header)
    range_txt = range_match.group(0)

    # 2. value (first numeric token NOT part of the range) --------------
    # remove the range fragment temporarily
    tmp = line.replace(range_txt, " ")
    value_match = VALUE_PAT.search(tmp)
    if not value_match:
        return None
    value_txt = value_match.group(1)

    # 3. unit (optional, appears after value or at end) -----------------
    unit_match = UNIT_PAT.search(tmp)
    unit_txt = unit_match.group(0) if unit_match else None

    # 4. test name = text before value token ----------------------------
    # find start index of value to slice left part
    value_pos = tmp.find(value_match.group(0))
    test_name = tmp[:value_pos].strip(" :-")
    if not test_name or len(test_name) < 2:
        return None

    # 5. calculate out-of-range flag ------------------------------------
    try:
        lo, hi = map(float, range_match.groups())
        val = float(value_txt)
        out_of_range = val < lo or val > hi
    except Exception:
        out_of_range = None  # cannot decide – keep None

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


# ----------------------------------------------------------------------
# -------------  FASTAPI ENDPOINT  -------------------------------------
# ----------------------------------------------------------------------
app = FastAPI(
    title="Lab-Report Extractor",
    description="Extracts lab test name, value, reference range & unit from a diagnostic report image",
    version="1.0.0",
)


@app.post("/get-lab-tests", response_model=LabResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    # --- validate mime type
    if file.content_type not in ("image/png", "image/jpeg", "application/pdf"):
        raise HTTPException(status_code=400, detail="Only PNG, JPEG or single-page PDF accepted")

    # --- save to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        with tmp_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        # If PDF, rasterise first page at 300dpi using OpenCV
        if tmp_path.suffix.lower() == ".pdf":
            try:
                # cv2.imdecode can't read pdf; use pdf2image if available
                from pdf2image import convert_from_path
                page = convert_from_path(str(tmp_path), dpi=300)[0]
                img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="pdf2image not installed: `pip install pdf2image`"
                )
        else:
            img = cv2.imdecode(np.fromfile(tmp_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image")

    # --- preprocessing + OCR
    processed = preprocess_image(img)
    lines = run_ocr(processed)

    # --- extraction
    lab_tests = extract_tests(lines)

    return JSONResponse(
        status_code=200,
        content=LabResponse(
            is_success=True,
            data={"lab_tests": [t.dict() for t in lab_tests]},
        ).dict(),
    )
