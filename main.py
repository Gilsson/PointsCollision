from os import path
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# from a2c import A2C
# from stable_baselines3 import A2C

# from my_env_points import envs
from tqdm import tqdm
import point_env
from torchvision.models.efficientnet import EfficientNet_B3_Weights
from rg_classifier import (
    CombinedNetwork,
    EfficientNetBasedMulticlass,
    HourglassBlock,
    MultipleStepClassifier,
    OneStepClassifier,
    CollisionClassifier,
    CustomDataset,
    OptunaNet,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.

n_envs = 2
n_updates = 10
n_steps_per_update = 128
randomize_domain = False
max_size = 10
# datasets_folder = "C:\\Users\\anton\\MachineLearning\\MyEnvPoints\\datasets"
datasets_folder = ".\\datasets"
# datasets_folder = "C:\\Users\\anton\\OneDrive\\Desktop\\WorkHorse"
add_folder = "128_14_4_train"
model_name = "first_attempt_step_1.pth"
# train_folder = "tempo"
train_folder = "30_points_4_radius_2_step_train_seed_45_square"
# train_folder = "128_14_4_validate_multiclass"
# val_folder = "128_14_4_test_multiclass_s"
val_folder = "30_points_4_radius_2_step_eval_seed_47_square"
optim_folder = "30_points_4_radius_2_step_test_seed_46_square"
# MODEL = CollisionClassifier(params)
# agent hyperparams
gamma = 0.999
lr = 0.003
batch_size = 10
weight_decay = 1e-5
num_epochs = 50
eval_every = 1
num_models = 10


def multiple_model_train(
    train_folder="32_15",
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-5,
    num_models=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = os.path.join(datasets_folder, train_folder)
    transform = v2.Compose(
        [
            v2.RandomRotation((35, 35)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomInvert(),
        ]
    )
    train_dataset = CustomDataset(root_dir=train_path, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=3)
    models = []
    models.append(CollisionClassifier().to(device))
    models.append(OneStepClassifier().to(device))
    models[1].load_state_dict(torch.load(".\\" + "trained_collision_classifier.pth"))
    models[0].load_state_dict(torch.load(".\\" + "trained_collision_classifier_3.pth"))
    # models[1].load_state_dict(torch.load(".\\" + "trained_collision_classifier.pth"))
    criterion = torch.nn.BCEWithLogitsLoss()
    [model.eval() for model in models]

    # Evaluation loop
    val_losses = []
    val_precisions = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for (images, labels), model in zip(data_loader, models):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

            val_losses.append(loss.item())

    precision = precision_score(
        np.array(all_labels), (np.array(all_outputs) > 0.5).astype(int)
    )
    val_precisions.append(precision)

    print(
        f"Validation Loss: {sum(val_losses) / len(data_loader)}, Precision: {precision}"
    )


def auc_roc_accuracy(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = os.path.join(datasets_folder, val_folder)
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    eval_dataset = CustomDataset(root_dir=train_path, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    if model == None:
        model = MODEL
        model = model.to(device)
        model.load_state_dict(torch.load(".\\" + model_name))
        model.eval()

    # Lists to store true labels and predicted scores
    true_labels = []
    predicted_scores = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in tqdm(eval_loader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predicted_labels.extend(
                (torch.sigmoid(outputs) > 0.3).float().cpu().numpy()
            )
            true_labels.extend(labels.cpu().numpy())
            predicted_scores.extend(probabilities.cpu().numpy())
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)
    auc_roc = roc_auc_score(true_labels, predicted_scores)
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    # plt.figure()
    # plt.plot(
    #     fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc_roc
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC curve")
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig("ROC_curve.png")
    # print(f"AUC ROC: {auc_roc}")
    return auc_roc


def confusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    val_path = os.path.join(datasets_folder, val_folder)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = CustomDataset(root_dir=val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    #  model = CollisionClassifier().to(device)
    model = MODEL.to(device)
    model_state_dict = torch.load(".\\" + model_name)
    model.load_state_dict(model_state_dict)
    model.eval()
    class_weights = [1187, 1966, 2935, 814, 615, 492, 424, 406, 360, 283, 286, 232]
    class_frequencies = [weight / 10000 for weight in class_weights]
    inverse_weights = [1 / freq for freq in class_frequencies]
    inverse_sum_weights = sum(inverse_weights)
    class_weights = [weight / inverse_sum_weights for weight in inverse_weights]
    weight = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    # criterion = f1_score

    # Evaluation loop
    val_losses = []
    val_precisions = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            all_labels.extend(label.item() for label in labels)
            predicitons = torch.argmax(outputs, dim=1)
            all_outputs.extend(prediction.item() for prediction in predicitons)

            val_losses.append(loss.item())
    cn_matrix = confusion_matrix(
        y_true=all_labels,
        y_pred=all_outputs,
        labels=[i for i in range(12)],
        normalize="true",
    )
    ConfusionMatrixDisplay(cn_matrix, display_labels=[str(i) for i in range(12)]).plot(
        include_values=False, xticks_rotation="vertical"
    )
    plt.title("Colors")
    plt.tight_layout()
    plt.show()


def recall_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_path = os.path.join(datasets_folder, val_folder)
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    val_loader = DataLoader(
        CustomDataset(root_dir=val_path, transform=transform), batch_size=batch_size
    )

    model = MODEL.to(device)
    model.load_state_dict(torch.load(".\\" + model_name))
    model.eval()

    with torch.no_grad():
        predicted_labels_list = []
        true_labels_list = []
        for images, labels in tqdm(val_loader, desc="Recall evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            predicted_labels_list.append(predicted_labels.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    predicted_labels_np = np.concatenate(predicted_labels_list)
    true_labels_np = np.concatenate(true_labels_list)
    score = recall_score(true_labels_np, predicted_labels_np)
    precision, recall, _ = precision_recall_curve(true_labels_np, predicted_labels_np)
    # plt.figure()

    # plt.plot(
    #     recall,
    #     precision,
    #     color="darkorange",
    #     lw=2,
    #     label="Precision-Recall curve (recall = %0.2f)" % score,
    # )
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")

    # plt.legend(loc="lower right")
    # plt.savefig("Recall.png")
    # plt.show()
    # print(f"Recall: {score}")

    # print(f"Precision: {precision[np.where(recall == score)[0][0]]}")
    return score


def train(model, params=None, epochs=None):
    num_epochs = epochs if epochs != None else 10
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data
    train_path = os.path.join(datasets_folder, train_folder)
    transform = v2.Compose(
        [
            v2.Resize((128, 128)),
            v2.RandomRotation((-30, 30)),
            # v2.GaussianBlur(kernel_size=3),  # Gaussian blur with kernel size 3
            v2.RandomInvert(),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = CustomDataset(root_dir=train_path, transform=transform)
    if params != None:
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    # model = MODEL.to(device)
    # models = [CollisionClassifier().to(device) for _ in range(num_models)]
    if not os.path.exists(".\\" + model_name):
        # If the file does not exist, create and save the model state dictionary
        torch.save(model.state_dict(), model_name)
        print(f"Model state dictionary saved to {model_name}")
    else:
        print(f"Model state dictionary already exists at {model_name}")
        model_state_dict = torch.load(".\\" + model_name)
        model.load_state_dict(model_state_dict)
    criterion = torch.nn.BCEWithLogitsLoss()
    if params != None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # Training loop
    train_losses = []
    train_precisions = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute precision
            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            # print(f"Predicted: {predicted_labels}")
            # print(f"Labels: {labels}")
            correct_predictions += (predicted_labels == labels).sum().item()

            total_samples += labels.size(0)

            running_loss += loss.item()
        print(f"Correct predictions: {correct_predictions}")
        epoch_loss = running_loss / len(train_loader)
        epoch_precision = correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_precisions.append(epoch_precision)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}, Precision: {epoch_precision}"
        )
        torch.save(model.state_dict(), model_name)
        # if (epoch + 1) % eval_every == 0:
        #     evaluate(model)

    return model


def train_multiclass(trained_model=None, params=None, epochs=None):
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
        root_dir=os.path.join(datasets_folder, train_folder), transform=transform
    )
    if params != None:
        data_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )
    else:
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # class_weights = [1864, 3062, 981, 767, 659, 532, 459, 417, 378, 320, 281, 280]
    # class_frequencies = [weight / 10000 for weight in class_weights]
    # inverse_weights = [1 / freq for freq in class_frequencies]
    # inverse_sum_weights = sum(inverse_weights)
    # class_weights = [weight / inverse_sum_weights for weight in inverse_weights]
    # weight = torch.tensor(dtype=torch.float).to(device)
    if trained_model == None:
        if params != None:
            model = MultipleStepClassifier(dropout=params["dropout"]).to(device)
        else:
            model = MODEL.to(device)
    else:
        model = trained_model
    if not os.path.exists(".\\" + model_name):
        # If the file does not exist, create and save the model state dictionary
        torch.save(model.state_dict(), model_name)

        def init_weights(m):
            import torch.nn.init as init

            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        model.apply(init_weights)
        print(f"Model state dictionary saved to {model_name}")
    else:
        if trained_model == None:
            model.load_state_dict(torch.load(".\\" + model_name))
            print(f"Model state dictionary already exists at {model_name}")

    criterion = torch.nn.CrossEntropyLoss()
    if params != None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if params != None:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params["factor"],
            patience=params["patience"],
            threshold=params["threshold"],
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    if epochs == None:
        epochs = num_epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_precision = 0
        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # # Compute precision
            # _, predicted_labels = torch.max(outputs, dim=1)
            # correct_predictions += (predicted_labels == labels).sum().item()
            # total_samples += labels.size(0)
            # epoch_precision = correct_predictions / total_samples
            running_loss += loss.item()
            # progress_bar.set_description(
            #     f"Epoch {epoch+1}/{num_epochs}, Precision {epoch_precision:.4f}"
            # )

        epoch_loss = running_loss / len(data_loader)
        # epoch_precision = correct_predictions / total_samples

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")
        torch.save(model.state_dict(), model_name)
        if (epoch + 1) % eval_every == 0:
            evaluate_multiclass(model)
    return model


def evaluate_multiclass(model=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    val_path = os.path.join(datasets_folder, val_folder)
    transform = v2.Compose(
        [
            v2.Resize((128, 128)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = CustomDataset(root_dir=val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    #  model = CollisionClassifier().to(device)
    if model == None:
        model = MODEL.to(device)
        model_state_dict = torch.load(".\\" + model_name)
        model.load_state_dict(model_state_dict)
    model.eval()
    # class_weights = [1864, 3062, 981, 767, 659, 532, 459, 417, 378, 320, 281, 280]
    # class_frequencies = [weight / 10000 for weight in class_weights]
    # inverse_weights = [1 / freq for freq in class_frequencies]
    # inverse_sum_weights = sum(inverse_weights)
    # class_weights = [weight / inverse_sum_weights for weight in inverse_weights]
    # weight = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = f1_score

    # Evaluation loop
    val_losses = []
    val_precisions = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_labels.extend(label.item() for label in labels)
            _, predictions = torch.max(outputs, dim=1)
            # print(predictions)
            all_outputs.extend(prediction.item() for prediction in predictions)

            val_losses.append(loss.item())

    recall = recall_score(all_labels, all_outputs, average=None)
    average_recall = recall_score(all_labels, all_outputs, average="macro")
    accuracy = accuracy_score(all_labels, all_outputs)

    print(
        f"Validation Loss: {sum(val_losses) / len(val_loader)}, Recall: {recall}\n, Accuracy : {accuracy}"
    )

    return average_recall, sum(val_losses) / len(val_loader)


def evaluate(model):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    val_path = os.path.join(datasets_folder, val_folder)
    transform = v2.Compose(
        [
            v2.Resize((128, 128)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = CustomDataset(root_dir=val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    #  model = CollisionClassifier().to(device)
    # model = MODEL.to(device)
    # model_state_dict = torch.load(model_name)
    # model.load_state_dict(model_state_dict)
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = f1_score

    # Evaluation loop
    val_losses = []
    val_precisions = []
    all_labels = []
    all_outputs = []
    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)
            # print(f"Predicted: {predicted_labels}")
            # print(f"Labels: {labels}")
            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            total_samples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            # all_labels.extend(labels.cpu().numpy())
            # all_outputs.extend((torch.sigmoid(outputs) > 0.5).float().cpu())
            val_losses.append(loss.item())

    print(f"Correct predictions: {correct_predictions}")
    # epoch_loss = running_loss / len(train_loader)
    epoch_precision = correct_predictions / total_samples
    # train_losses.append(epoch_loss)
    # train_precisions.append(epoch_precision)

    precision = precision_score(
        np.array(all_labels), (np.array(all_outputs)).astype(int)
    )
    val_precisions.append(precision)
    # recall = recall_score(all_labels, all_outputs, average=None)
    # average_recall = recall_score(all_labels, all_outputs, average="macro")

    print(
        f"Validation Loss: {sum(val_losses) / len(val_loader)}, Precision: {epoch_precision}"
    )

    return precision


def hypertuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_grid = {
        "learning_rate": [
            1e-8,
            1e-7,
            1e-6,
            5e-5,
            1e-5,
            5e-4,
            1e-4,
            5e-3,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
        ],
        "weight_decay": [
            1e-6,
            1e-7,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            5e-2,
            5e-3,
            5e-4,
        ],
        "batch_size": [10, 20, 40, 50, 80, 100, 120],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "threshold": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-2, 5e-3, 5e-4],
        "patience": [1, 5, 10, 20],
        # Add other hyperparameters as needed
    }
    best_average_recall = 0
    best_params = None

    for _ in range(200):  # Number of random search iterations
        params = {
            "learning_rate": np.random.choice(param_grid["learning_rate"]),
            "weight_decay": np.random.choice(param_grid["weight_decay"]),
            "batch_size": int(np.random.choice(param_grid["batch_size"])),
            "dropout": np.random.choice(param_grid["dropout_rate"]),
            "factor": np.random.choice(param_grid["factor"]),
            "threshold": np.random.choice(param_grid["threshold"]),
            "patience": np.random.choice(param_grid["patience"]),
            # Add other hyperparameters as needed
        }
        # model = EfficientNetBasedMulticlass(
        #     dropout=params["dropout"],
        #     weights=EfficientNet_B3_Weights.IMAGENET1K_V1,
        #     num_classes=3,
        # ).to(device)
        model = CombinedNetwork().to(device)
        # model = MultipleStepClassifier(dropout=params["dropout"]).to(device)
        # def init_weights(m):
        #     import torch.nn.init as init

        #     if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        #         init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

        # model.apply(init_weights)
        model = train_multiclass(model, params=params, epochs=1)
        average_recall, loss = evaluate_multiclass(model)

        # while loss > 10 and abs(accuracy - best_accuracy) < 1e-5:
        #     model = train_multiclass(model, params, epochs=1)
        #     accuracy, loss = evaluate_multiclass(model)
        print(
            f"Average Recall: {average_recall}, Hyperparameters: {params}, Loss: {loss}"
        )
        if average_recall > best_average_recall:
            best_average_recall = average_recall
            best_params = params

    print("Best Accuracy:", best_average_recall)
    print("Best Hyperparameters:", best_params)


if __name__ == "__main__":
    # point_env.generate_dataset_multiclass(
    #     size=3000,
    #     n_points=30,
    #     # max_image_shape=(50, 50),
    #     points_max_radius=4,
    #     steps=1,
    #     name="30_points_4_radius_2_step_eval_seed_47_square",
    #     path=datasets_folder,
    #     remove_existing_folder=True,
    #     seed=47,
    # )
    # evaluate_multiclass()
    # confusion()
    # precision_multiclass()
    # hypertuning()
    params = {
        "learning_rate": 0.001,
        "weight_decay": 1e-06,
        "batch_size": 80,
        "dropout": 0.6,
        "factor": 0.8,
        "threshold": 0.1,
        "patience": 5,
    }

    # # # # # # # evaluate_multiclass()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = EfficientNetBasedMulticlass(
    #     num_classes=3,
    #     dropout=params["dropout"],
    #     weights=EfficientNet_B3_Weights.DEFAULT,
    # ).to(device)
    # model = CollisionClassifier(params).to(device)
    # model = MultipleStepClassifier(3, params["dropout"]).to(device)
    num_layers = 16
    num_filters = [
        30,
        40,
        60,
        80,
        97,
        133,
        111,
        75,
        181,
        104,
        192,
        182,
        141,
        160,
        180,
        200,
    ]
    use_maxpool = [
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    model = OptunaNet(num_layers, num_filters, use_maxpool).to(device)
    # model.load_state_dict(torch.load(model_name))
    train_multiclass(model, params, 100)
    # evaluate(model)
    # model = train(model, params, 100)
    # model = train_multiclass(model, params, 10)
    # model = MultipleStepClassifier(params).to(device)
    # model.load_state_dict(torch.load(model_name))
    # model = MultipleStepClassifier(dropout=params["dropout"]).to(device)
    # model = train_multiclass(model, params, 100)
    # model.load_state_dict(torch.load(".\\" + model_name))
    # evaluate_multiclass(model)
