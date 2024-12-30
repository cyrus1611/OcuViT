
# Training Function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, dtype=torch.float32), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100. * correct / total
    return total_loss / len(loader), accuracy

# Evaluation Function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100. * correct / total
    return total_loss / len(loader), accuracy

# Logging Training and Evaluation Results
def log_results_to_csv(epoch, train_loss, train_accuracy, test_loss, test_accuracy, filename="training_log.csv"):
    header = ["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]
    try:
        with open(filename, "x") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except FileExistsError:
        pass

    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_accuracy, test_loss, test_accuracy])

# Metrics Calculation and Confusion Matrix Plotting
def calculate_metrics_and_plot_confusion_matrix(model, loader, device, confusion_matrix_path="confusion_matrix.png"):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(
            torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=5),
            torch.nn.functional.softmax(torch.tensor(outputs), dim=1),
            multi_class='ovr'
        )
    except ValueError:
        auc = None  # AUC cannot be calculated if there are missing classes

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%)')
    plt.savefig(confusion_matrix_path)
    plt.show()

# Early Stopping Parameters
patience = 10
best_test_accuracy = 0.0
best_metrics = {"precision": 0.0, "recall": 0.0, "auc": 0.0}
counter = 0

# Training and Evaluation Loop with Logging and Early Stopping
epochs = 50
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Training
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Log metrics to CSV
    log_results_to_csv(epoch + 1, train_loss, train_accuracy, test_loss, test_accuracy)

    # Early Stopping Logic and Metric Tracking
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        counter = 0

        # Calculate additional metrics
        y_true = []
        y_pred = []

        # Collect predictions for metrics
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(images).logits
                _, predicted = outputs.max(1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        best_metrics["precision"] = precision_score(y_true, y_pred, average='weighted')
        best_metrics["recall"] = recall_score(y_true, y_pred, average='weighted')
        try:
            best_metrics["auc"] = roc_auc_score(
                torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=5),
                torch.nn.functional.softmax(torch.tensor(outputs), dim=1),
                multi_class='ovr'
            )
        except ValueError:
            best_metrics["auc"] = None

        # Save the best model
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved as best_model.pth")
    else:
        counter += 1
        print(f"Early stopping counter: {counter}/{patience}")

    if counter >= patience:
        print("Early stopping triggered.")
        break

# Final Results Output
print("\nBest Test Accuracy and Metrics:")
print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
if best_metrics['auc'] is not None:
    print(f"AUC: {best_metrics['auc']:.4f}")
else:
    print("AUC: Not calculated (missing classes)")

# Final Evaluation and Metrics
calculate_metrics_and_plot_confusion_matrix(model, test_loader, device, confusion_matrix_path="confusion_matrix.png")

# Load the best model for evaluation
model.load_state_dict(torch.load("best_model.pth"))
print("Loaded the best model for final evaluation.")
