import string
#
# import cv2
# import easyocr
# import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
# from ultralytics import YOLO

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

# Initialize models
# coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')
# reader = easyocr.Reader(['en'], gpu=False)

# List of vehicle class IDs from COCO dataset that are considered
vehicles = [2, 3, 5, 7]

# Helper functions
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {v: k for k, v in dict_char_to_int.items()}

@app.get("/")
def read_root():
    return {"Hello": "World"}

# def license_complies_format(text):
#     if len(text) != 7:
#         return False
#
#     if all((c in string.ascii_uppercase or c in dict_int_to_char) for c in text[:2] + text[4:]) and \
#             all(c.isdigit() or c in dict_char_to_int for c in text[2:4]):
#         return True
#     return False


# def format_license(text):
#     license_plate_ = ''
#     for i, c in enumerate(text):
#         if i in [2, 3] and c in dict_char_to_int:
#             license_plate_ += dict_char_to_int[c]
#         elif c in dict_int_to_char:
#             license_plate_ += dict_int_to_char[c]
#         else:
#             license_plate_ += c
#     return license_plate_
#
#
# def read_license_plate(license_plate_crop):
#     detections = reader.readtext(license_plate_crop)
#     for bbox, text, score in detections:
#         text = text.upper().replace(' ', '')
#         if license_complies_format(text):
#             return format_license(text), score
#     return None, None
#
#
# def process_image(frame):
#     vehicle_detections = coco_model(frame)[0]
#     vehicle_detections = [d for d in vehicle_detections.boxes.data.tolist() if int(d[5]) in vehicles]
#
#     results = []
#
#     if vehicle_detections:
#         for vehicle in vehicle_detections:
#             x1, y1, x2, y2 = map(int, vehicle[:4])
#             vehicle_crop = frame[y1:y2, x1:x2]
#
#             if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
#                 continue
#
#             license_plates = license_plate_detector(vehicle_crop)[0]
#             for license_plate in license_plates.boxes.data.tolist():
#                 lp_x1, lp_y1, lp_x2, lp_y2, lp_score, _ = license_plate
#                 license_plate_crop = vehicle_crop[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
#                 if license_plate_text:
#                     results.append(
#                         {'bbox': [lp_x1 + x1, lp_y1 + y1, lp_x2 + x1, lp_y2 + y1], 'text': license_plate_text,
#                          'score': license_plate_text_score})
#
#     return results

#
# @app.post("/detect-license-plates/")
# async def detect_license_plates(file: UploadFile, location: str = "local"):
#     image_bytes = await file.read()
#     image = np.frombuffer(image_bytes, dtype=np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#
#     license_plates = process_image(image)
#     if not license_plates:
#         return {"license_plates": ""}
#     else:
#         return {"license_plates": license_plates[0]}
