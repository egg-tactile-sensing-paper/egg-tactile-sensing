if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO('yolov8m.yaml')
    model.train(data='config.yaml', epochs=350)
