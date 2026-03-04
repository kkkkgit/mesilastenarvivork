import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Kasutan seadet: {device}")

    print("Laen mudeleid...")
    yolo_model = YOLO('yolov8n.pt')
    sam_model = SAM('sam2_t.pt')
    print("Mudelid laetud!")

    # YOLO class ID 0 = 'person'
    PERSON_CLASS_ID = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ei saa kaamerat avada")
        exit()

    print("\nSPACE = tuvasta ja lõika inimesed välja | Q = välju\n")

    annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = annotated if annotated is not None else frame
        cv2.imshow("Inimeste segmenteerimine", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            print("Tuvastan inimesi...")

            # Step 1: YOLO finds people
            yolo_results = yolo_model(frame, verbose=False)
            boxes = yolo_results[0].boxes

            # Filter only 'person' class
            person_mask = boxes.cls.cpu().numpy().astype(int) == PERSON_CLASS_ID
            person_boxes = boxes.xyxy.cpu().numpy()[person_mask].tolist()

            if len(person_boxes) == 0:
                print("Inimesi ei tuvastatud")
                annotated = None
                continue

            print(f"Tuvastatud {len(person_boxes)} inimest, segmenteerin...")

            # Step 2: SAM2 segments each person precisely
            sam_results = sam_model(frame, bboxes=person_boxes, verbose=False)

            # Step 3: Create output with only people (black background)
            # people_only = np.zeros_like(frame)
            people_only = cv2.GaussianBlur(frame, (51, 51), 0)
            annotated_overlay = frame.copy()

            if sam_results and sam_results[0].masks is not None:
                masks = sam_results[0].masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    mask_bool = mask.astype(bool)

                    # Copy person pixels to black background
                    people_only[mask_bool] = frame[mask_bool]

                    # Draw colored overlay on original
                    color = [0, 255, 128]  # green overlay
                    overlay = annotated_overlay.copy()
                    overlay[mask_bool] = color
                    annotated_overlay = cv2.addWeighted(
                        annotated_overlay, 0.7, overlay, 0.3, 0
                    )

                    # Draw contour
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated_overlay, contours, -1, (255, 255, 255), 2)

                print(f"{len(masks)} inimest segmenteeritud!")

            # Show both views side by side
            combined = np.hstack([annotated_overlay, people_only])
            annotated = combined

        elif key == ord('c'):
            annotated = None
            print("Puhastatud")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()