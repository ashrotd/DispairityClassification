

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


manual_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,),(0.5,)),
        
    transforms.RandomHorizontalFlip(p=0.5)
])
dataset = torchvision.datasets.ImageFolder(root="C:/disparity_classification_dataset",transform=manual_transform)


dataset[1]


def label_to_classname(label, dataset):
    return dataset.classes[label]

# Example usage
image, label = dataset[4000]
print(image, label_to_classname(label, dataset))


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# Checking some training images
images, labels = next(iter(train_loader))
print("Image batch dimensions:", images[0].shape)
print("Image label dimensions:", labels.shape)


for x,y in train_loader:
    plt.imshow(x[23].permute(2,1,0))
    print(x[23].shape)
    break

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_classes = len(dataset.classes)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)


# In[12]:


import matplotlib.pyplot as plt

num_epochs = 10  # Set the number of epochs


train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()  # Backward pass and optimize
        optimizer.step()

        running_loss += loss.item()
    
    # Calculate average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation after each epoch
    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    # Print epoch summary
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {avg_train_loss:.3f}')
    print(f'Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

print('Finished Training')

# Plotting
plt.figure(figsize=(12, 4))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


from PIL import Image
import cv2

test_dir = 'C:/disparity_dataset/image_100.png'
grayscale_image = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)

# Convert the grayscale image to 3-channel RGB
rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
# Convert the numpy array to PIL image for compatibility with torchvision transforms
rgb_image = Image.fromarray(rgb_image)
rgb_image_tensor = manual_transform(rgb_image)
rgb_image_tensor.shape


model.eval()  # Set model to evaluation mode
transformed_image = rgb_image_tensor.to(device)  # Move to device

with torch.no_grad():
    output = model(rgb_image_tensor.to(device))


r,t=torch.max(output,1)
t

test_dataset = torchvision.datasets.ImageFolder(root="C:/disparity_classification_dataset_test",transform=manual_transform)
test_loader = DataLoader(dataset=test_dataset,batch_size=4,shuffle=False)



for x,y in test_loader:
    plt.imshow(x[3].permute(2,1,0))
    break

# Evaluate the model performance bases on the testing images
model.eval()
correct = 0
total = 0

all_prediction = []
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        all_prediction.extend(predicted.cpu().numpy())
test_accuracy = 100*correct/total
print(f'Accuracy on test data: {test_accuracy}%')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get all true labels
true_labels = [label for _, label in test_dataset]

# Compute confusion matrix
cm = confusion_matrix(true_labels, all_prediction)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Dataset visualization
import os
root_dir = 'C:/disparity_classification_dataset/'

count_classes = {}

for folders in os.listdir(root_dir):
    class_path = os.path.join(root_dir,folders)
    if os.path.isdir(class_path):
        count_classes[folders] = len(os.listdir(class_path))

print(count_classes)


import seaborn as sns

sns.barplot(x=list(count_classes.keys()), y=list(count_classes.values()), palette='viridis')
plt.xlabel('Training classes in meters')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.xticks(rotation=45)  # Rotate class names for better readability if needed
plt.tight_layout()  # Adjust layout to fit class names
plt.show()


test_dirs = 'C:/disparity_classification_dataset_test/'

count_tests = {}

for folders in os.listdir(test_dirs):
    class_path = os.path.join(test_dirs,folders)
    
    if os.path.isdir(class_path):
        count_tests[folders] = len(os.listdir(class_path))
print(count_tests)


sns.barplot(x=list(count_tests.keys()),y=count_tests.values(), palette='viridis')
plt.xlabel('Test classes range in meters')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.xticks(rotation=45)  # Rotate class names for better readability if needed
plt.tight_layout()  # Adjust layout to fit class names
plt.show()



