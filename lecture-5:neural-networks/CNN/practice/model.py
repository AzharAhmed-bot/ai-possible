import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil, glob
import json



def get_label_to_name_dict(save_path="label_to_name.json"):
    # If file already exists, just load from it
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            label_to_name = json.load(f)
        return {int(k): v for k, v in label_to_name.items()}  # keys back to int

    # Otherwise, fetch dataset and create mapping
    ds = load_dataset("Soumyajit9979/animals-dataset")
    label2name = ds['train'].features['label'].int2str
    unique_labels = set(ds['train']['label'])
    label_to_name = {label: label2name(label) for label in unique_labels}

    # Save mapping for future runs
    with open(save_path, "w") as f:
        json.dump(label_to_name, f)

    return label_to_name

# =========================================================
# 1. Download & Save Dataset
# =========================================================
def save_hf_animals_dataset(root_dir="data", max_per_class=20):
    ds = load_dataset("Soumyajit9979/animals-dataset")
    split_dir = os.path.join(root_dir, "train")
    os.makedirs(split_dir, exist_ok=True)
    class_counts = {}

    for i, item in enumerate(ds["train"]):
        label = item["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
        if class_counts[label] > max_per_class:
            continue
        img = item["image"]
        label_dir = os.path.join(split_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f"train_{i}.jpg")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img.save(img_path)
        if i < 3:
            print(f"Saved: {img_path}")


# =========================================================
# 2. Train/Test Split
# =========================================================
def split_data(root_dir="data", test_size=0.2):
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    for label in os.listdir(train_dir):
        files = glob.glob(os.path.join(train_dir, label, "*.jpg"))
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
        test_label_dir = os.path.join(test_dir, label)
        os.makedirs(test_label_dir, exist_ok=True)
        for f in test_files:
            shutil.move(f, test_label_dir)
    print("Dataset split into train/ and test/.")


# =========================================================
# 3. Data Loaders
# =========================================================
def get_data_loaders(train_dir, test_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_data.classes



# =========================================================
# 4. Build Model
# =========================================================
def build_model(num_classes):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")  # pretrained on ImageNet
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


# =========================================================
# 5. Train Model
# =========================================================
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


# =========================================================
# 6. Test on Webcam
# =========================================================
def load_classes_from_json(json_file="label_to_name.json"):
    with open(json_file, "r") as f:
        return json.load(f)

import torch.nn.functional as F

def test_model(model, device, json_file="label_to_name.json"):
    model.to(device)
    model.eval()

    classes = load_classes_from_json(json_file)

    cap = cv.VideoCapture('/dev/video0', cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
            probs = F.softmax(output, dim=1)
            conf, predicted = torch.max(probs, 1)
            label = str(predicted.item())
            animal_name = classes.get(label, "Unknown")
            confidence = conf.item() * 100

        cv.putText(frame, f"{animal_name} ({confidence:.1f}%)", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Animal Classifier", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()




# =========================================================
# 7. Main
# =========================================================
def main():
    root_dir = "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(root_dir):
        save_hf_animals_dataset(root_dir)
        split_data(root_dir)
    # Data loaders (only needed for training OR to get the class labels)
    train_loader, test_loader, classes = get_data_loaders(
        os.path.join(root_dir, "train"),
        os.path.join(root_dir, "test")
    )

    model = build_model(len(classes))

    if os.path.exists("animal_classifier.pth"):
        print("Loading existing model...")
        model.load_state_dict(torch.load("animal_classifier.pth", map_location=device))
    else:
        print("No saved model found, training a new one...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer, device, epochs=5)
        torch.save(model.state_dict(), "animal_classifier.pth")
        print("Model trained and saved!")

    # Run on webcam
    test_model(model, device)



if __name__ == "__main__":
    main()
