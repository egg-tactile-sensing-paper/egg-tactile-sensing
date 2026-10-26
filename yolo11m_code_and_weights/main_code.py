if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO('yolo11m.yaml')
    model.train(data='config.yaml', epochs=350)
