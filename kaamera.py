import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image


# Must define the class so torch.load can reconstruct it
class BeeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Linear(16 * 23 * 76, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, data):
        out = self.conv1(data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    my_model = torch.load("model.pkl", map_location=device, weights_only=False)
    my_model.eval()
    print("Mudel laetud!")

    # Same classes as training data
    class_names = ['bee', 'othr', 'wasp']  # alphabetical, same as ImageFolder

    # Preprocessing to match training
    preprocess = transforms.Compose([
        transforms.Resize((350, 600)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ei saa kaamerat avada")
        exit()

    print("SPACE = pildista ja klassifitseeri | Q = välju")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Mesilaste tuvastus", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            test_tensor = preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = my_model(test_tensor)
                probs = F.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)

            label = class_names[pred_idx.item()]
            score = confidence.item()
            print(f"Tuvastus: {label} | Kindlus: {score*100:.1f}%")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()