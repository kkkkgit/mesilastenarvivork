import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLOv8 nano model (fast, good for real-time)
    # Downloads automatically on first run (~6MB)
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ei saa kaamerat avada")
        exit()

    print("Objektituvastus käivitatud! Q = välju")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame, verbose=False)

        # Draw bounding boxes on frame
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 objektituvastus", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()