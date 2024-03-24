from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import v8
import math
import cv2
import easyocr
import numpy as np
from deskew import determine_skew

app = FastAPI()
handler = Mangum(app)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['en'], gpu=False)


def preprocess_image(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.GaussianBlur(img, (1, 1), 0)
    return smooth


def deskew_plate(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    # print(angle)
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(0, 0, 0))


def text(img):
    try:
        text = ""
        for ele in reader.readtext(img, allowlist='.-0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'):
            text = text + str(ele[1])
        print("OCR Success!")
        return text
    except:
        print("OCR Failed!")
        return " "


def detect_objects(img, conf, model_type):
    model = v8.ANPR_V8("best.pt")
    plates, image = model.detect(img, conf)
    return image, plates


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect-license-plates/")
async def detect_license_plates(file: UploadFile, location: str = "local"):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    uploaded_file = cv2.imdecode(image, cv2.IMREAD_COLOR)
    conf = 0.5
    model_type = "v8"
    # uploaded_file = cv2.imread("test_ocr.jpg")

    if uploaded_file is not None:
        is_success, im_buf_arr = cv2.imencode(".jpg", uploaded_file)
        byte_im = im_buf_arr.tobytes()
        file_bytes = np.frombuffer(byte_im, dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = deskew_plate(image)
        output_image, plates = detect_objects(image, conf, model_type)
        if len(plates):
            for plate in plates:
                x1, y1, x2, y2, conf = plate
                crop_img = output_image[y1:y2, x1:x2]
                processed_img = preprocess_image(crop_img)
                result = text(processed_img)
                print(result)

                print("number plate detected")

                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                cv2.putText(output_image, f"{result}", (x1 + 7, (y1 + y2) // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 3, cv2.LINE_AA)

                return {"license_plate": result}
        else:
            print("No License Plate Detected")
            return {"license_plate": "No License Plate Detected"}
