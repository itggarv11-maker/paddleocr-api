from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import paddleocr

app = FastAPI()

ocr = paddleocr.OCR(use_angle_cls=True, lang='en')  # Set lang='en' or 'en+hindi'

def read_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.post("/ocr")
async def run_paddle_ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image(image_bytes)

    result = ocr.ocr(image, cls=True)
    texts = [line[1][0] for block in result for line in block]

    return JSONResponse(content={"text": "\n".join(texts)})
