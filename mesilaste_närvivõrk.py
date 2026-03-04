import torch
from torch import nn
import urllib.request, os, zipfile

# --- Imports used later (moved to top for clarity) ---
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import requests
from PIL import Image
import io


# =====================================================================
# Define the network class BEFORE the main guard
# =====================================================================
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


# =====================================================================
# Everything below MUST be inside the main guard for macOS
# =====================================================================
if __name__ == '__main__':

    # -----------------------------------------------------------------
    # 1. Download and extract data
    # -----------------------------------------------------------------
    if not os.path.exists("bee3.zip"):
        print("Downloading bee dataset...")
        urllib.request.urlretrieve("http://linuxator.com/data/bee3.zip", "bee3.zip")
    os.makedirs("bee_new", exist_ok=True)
    with zipfile.ZipFile("bee3.zip", "r") as z:
        z.extractall("bee_new")

    # -----------------------------------------------------------------
    # 2. Prepare dataset and dataloader
    # -----------------------------------------------------------------
    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    bee_dataset = datasets.ImageFolder(root='bee_new', transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(
        bee_dataset, batch_size=4, shuffle=True, num_workers=0  # num_workers=0 for macOS
    )

    # -----------------------------------------------------------------
    # 3. Init model, device, loss, optimizer
    # -----------------------------------------------------------------
    model = BeeNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_crit = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------------------------------------------
    # 4. Training loop (20 epochs)
    # -----------------------------------------------------------------
    num_epoch = 20
    for epoch in range(num_epoch):
        epoch_loss = 0
        model.train()
        for i, (img, labels) in enumerate(dataset_loader):
            img, labels = img.to(device), labels.to(device)
            optimiser.zero_grad()
            predictions = model(img)
            loss = loss_crit(predictions, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimiser.step()

        epoch_loss = epoch_loss / (i + 1)
        print(f"Loss at epoch {epoch} is {epoch_loss:.4f}")

    # -----------------------------------------------------------------
    # 5. Evaluate on training data (overfit check)
    # -----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for img, label in dataset_loader:
            img, label = img.to(device), label.to(device)
            prediction = model(img)
            print(prediction)
            print("LABEL", label)
            pred_class = torch.max(prediction, dim=1)
            print("CLASS", pred_class.indices)

    # -----------------------------------------------------------------
    # 6. Save and reload model
    # -----------------------------------------------------------------
    torch.save(model, "model.pkl")
    my_model = torch.load("model.pkl", map_location=device, weights_only=False)

    # -----------------------------------------------------------------
    # 7. Show bee predictions from training data
    # -----------------------------------------------------------------
    bee_class_index = bee_dataset.class_to_idx['bee']
    max_to_show = 5
    found_count = 0

    my_model.eval()
    with torch.no_grad():
        for images, labels in dataset_loader:
            images = images.to(device)
            outputs = my_model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                if predictions[i] == bee_class_index:
                    found_count += 1
                    img_np = images[i].squeeze().cpu().numpy()
                    score = confidences[i].item()
                    plt.figure(figsize=(4, 4))
                    plt.imshow(img_np, cmap='gray')
                    plt.title(f"Class: Bee | Score: {score:.4f} ({score*100:.2f}%)")
                    plt.axis('off')
                    plt.show()
                if found_count >= max_to_show:
                    break
            if found_count >= max_to_show:
                break

    # -----------------------------------------------------------------
    # 8. Download external honeybee image and classify
    # -----------------------------------------------------------------
    url = "https://cdn-ifjcn.nitrocdn.com/VSJPVATxenzNJTzhEwVMsOyeLsByReHl/assets/images/optimized/rev-054c3dd/beegone.co.uk/wp-content/uploads/2024/03/Honeybee-scaled.webp"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    test_image = Image.open(io.BytesIO(response.content))
    print(f"Downloaded image. Format: {test_image.format}, Size: {test_image.size}")

    # Get training image dimensions for correct resize
    train_img, _ = bee_dataset[0]
    train_h, train_w = train_img.shape[1], train_img.shape[2]
    print(f"Training image size: {train_w}x{train_h}")

    preprocess = transforms.Compose([
        transforms.Resize((train_h, train_w)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    test_image_tensor = preprocess(test_image).unsqueeze(0).to(device)

    my_model.eval()
    with torch.no_grad():
        logits = my_model(test_image_tensor)

    probabilities = F.softmax(logits, dim=1)
    conf, pred_idx = torch.max(probabilities, dim=1)
    class_names = bee_dataset.classes
    predicted_label = class_names[pred_idx.item()]
    confidence_score = conf.item()

    plt.figure(figsize=(6, 6))
    plt.imshow(test_image)
    plt.title(f"Prediction: {predicted_label} | Confidence: {confidence_score:.4f} ({confidence_score*100:.2f}%)")
    plt.axis('off')
    plt.show()
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence_score:.4f}")

    # -----------------------------------------------------------------
    # 9. Visualize training bee images
    # -----------------------------------------------------------------
    bee_dir = 'bee_new/bee'
    bee_files = sorted([f for f in os.listdir(bee_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])[:12]

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('Training Samples: Bee Class', fontsize=16)
    for i, ax in enumerate(axes.flat):
        if i < len(bee_files):
            img_path = os.path.join(bee_dir, bee_files[i])
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(bee_files[i])
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # -----------------------------------------------------------------
    # 10. Retrain with data augmentation
    # -----------------------------------------------------------------
    augmented_transform = transforms.Compose([
        transforms.Resize((350, 600)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(size=(350, 600), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    bee_dataset = datasets.ImageFolder(root='bee_new', transform=augmented_transform)
    dataset_loader = torch.utils.data.DataLoader(
        bee_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    print(f"Dataset updated with augmentation. Samples: {len(bee_dataset)}")
    print(f"Classes: {bee_dataset.classes}")

    # Re-init model and retrain
    model = BeeNet().to(device)
    loss_crit = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting augmented training on {device} for {num_epoch} epochs...")
    for epoch in range(num_epoch):
        epoch_loss = 0
        model.train()
        for i, (img, labels) in enumerate(dataset_loader):
            img, labels = img.to(device), labels.to(device)
            optimiser.zero_grad()
            predictions = model(img)
            loss = loss_crit(predictions, labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / (i + 1)
        print(f"Epoch [{epoch+1}/{num_epoch}] - Loss: {avg_epoch_loss:.4f}")

    my_model = model
    print('Retraining completed.')

    # -----------------------------------------------------------------
    # 11. Re-evaluate external image with retrained model
    # -----------------------------------------------------------------
    val_preprocess = transforms.Compose([
        transforms.Resize((350, 600)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    test_image_tensor = val_preprocess(test_image).unsqueeze(0).to(device)

    my_model.eval()
    with torch.no_grad():
        logits = my_model(test_image_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)

    predicted_label = bee_dataset.classes[pred_idx.item()]
    score = confidence.item()

    plt.figure(figsize=(6, 6))
    plt.imshow(test_image)
    plt.title(f"Retrained Prediction: {predicted_label} | Confidence: {score:.4f} ({score*100:.2f}%)")
    plt.axis('off')
    plt.show()
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence Score: {score:.4f}")