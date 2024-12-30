from transformers import ViTForImageClassification

# Model Definition
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=5,  # Updated for 5-class classification
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
