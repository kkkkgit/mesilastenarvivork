import cv2
import numpy as np
import torch
from ultralytics import SAM

# Store mouse clicks
click_points = []
click_labels = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click = positive point
        click_points.append([x, y])
        click_labels.append(1)
        print(f"+ Punkt lisatud: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click = negative point
        click_points.append([x, y])
        click_labels.append(0)
        print(f"- Negatiivne punkt: ({x}, {y})")


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Kasutan seadet: {device}")

    print("Laen SAM2 mudelit...")
    sam_model = SAM('sam2_t.pt')  # tiny = kiire, 'sam2_b.pt' = täpsem
    print("Mudel laetud!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ei saa kaamerat avada")
        exit()

    window_name = "SAM2 segmenteerimine"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nJuhised:")
    print("  Vasak klikk  = lisa punkt (segmenteeri see objekt)")
    print("  Parem klikk  = negatiivne punkt (välista see ala)")
    print("  SPACE        = jooksuta segmenteerimine")
    print("  C            = puhasta punktid")
    print("  Q            = välju\n")

    current_frame = None
    annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame.copy()
        display = annotated if annotated is not None else frame.copy()

        # Draw click points on display
        for i, pt in enumerate(click_points):
            color = (0, 255, 0) if click_labels[i] == 1 else (0, 0, 255)
            cv2.circle(display, (pt[0], pt[1]), 6, color, -1)
            cv2.circle(display, (pt[0], pt[1]), 8, (255, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and len(click_points) > 0:
            print("Segmenteerin...")

            results = sam_model(
                current_frame,
                points=click_points,
                labels=click_labels,
                verbose=False
            )

            annotated = current_frame.copy()
            if results and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                for mask in masks:
                    color = np.random.randint(100, 255, 3).tolist()
                    mask_bool = mask.astype(bool)
                    overlay = annotated.copy()
                    overlay[mask_bool] = color
                    annotated = cv2.addWeighted(annotated, 0.5, overlay, 0.5, 0)

                    # Draw mask contour
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated, contours, -1, (255, 255, 255), 2)

                print(f"Valmis! Tuvastatud {len(masks)} maski")
            else:
                print("Maske ei leitud")

        elif key == ord('c'):
            click_points.clear()
            click_labels.clear()
            annotated = None
            print("Punktid puhastatud")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()