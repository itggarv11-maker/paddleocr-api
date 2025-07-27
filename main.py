from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import shutil

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='en')


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
