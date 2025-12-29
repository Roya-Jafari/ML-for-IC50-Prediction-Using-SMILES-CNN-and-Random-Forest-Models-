# ----------------------------
# CNN Model
# ----------------------------

class SMILESCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs, padding="same")
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [self.relu(conv(x)).max(dim=2)[0] for conv in self.convs]  # list of tensors
        x = torch.cat(x, dim=1)      # convert list -> tensor
        x = self.dropout(x)          # dropout
        return self.fc(x).squeeze(1)

# ----------------------------
# Hyperparameters
# ----------------------------
EMBED_DIM = 512
NUM_FILTERS = 1024
FILTER_SIZES = [5, 7]
EPOCHS = 200

# ----------------------------
# Training setup
# ----------------------------
model = SMILESCNN(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_filters=NUM_FILTERS,
    filter_sizes=FILTER_SIZES
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
r2_scores = []
# ----------------------------
# Training loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # R² on training set
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(p.cpu().numpy())

    r2 = r2_score(y_true, y_pred)
    r2_scores.append(r2)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss={avg_loss:.4f} | R²={r2:.4f}")

# --- EVALUATE ON TEST SET AFTER TRAINING ---
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        p = model(x).view(-1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(p.cpu().numpy())

from sklearn.metrics import r2_score, mean_squared_error
print("Test R²:", r2_score(y_true, y_pred))
print("Test MSE:", mean_squared_error(y_true, y_pred))
