from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['image.jpg'])
print(results)