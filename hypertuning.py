import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import optuna
from rg_classifier import CustomDataset, OptunaNet
from torchvision.transforms import v2
from torch.utils.data import DataLoader

datasets_folder = ".\\datasets"
model_name = "first_attempt_step_1.pth"
# train_folder = "tempo"
train_folder = "30_points_4_radius_2_step_train_seed_45_square"
val_folder = "30_points_4_radius_2_step_eval_seed_47_square"
optim_folder = "30_points_4_radius_2_step_test_seed_46_square"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = v2.Compose(
    [
        v2.Resize((128, 128)),
        v2.RandomRotation((-60, 60)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CustomDataset(
    root_dir=os.path.join(datasets_folder, val_folder), transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)


def objective(trial):
    # Suggest hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 12)
    num_filters = [
        trial.suggest_int(f"num_filters_{i}", 4, 200) for i in range(num_layers)
    ]
    use_maxpool = [
        trial.suggest_categorical(f"use_maxpool_{i}", [True, False])
        for i in range(num_layers)
    ]

    # Create the model
    model = OptunaNet(num_layers, num_filters, use_maxpool).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(3):  # Use a small number of epochs for the example
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy


# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)

print("Best trial:")
trial = study.best_trial
print("  Accuracy: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
