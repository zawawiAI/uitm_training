from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/zawawi/Documents/unimas/cili.v1i.yolov11/data.yaml", epochs=100 )
