from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Sample data (replace with golden nuget dataset)
texts = ["I love this product!", "This is terrible."]
labels = [1.0, -1.0]  # Continuous sentiment scores

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Split the data into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs['input_ids'], torch.tensor(labels), test_size=0.2)
train_data = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(test_inputs, test_labels)
test_loader = DataLoader(test_data, batch_size=32)
# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.train()

# Define loss and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs).logits.squeeze()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
model.eval()
total_loss = 0

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs).logits.squeeze()
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

print("Test Loss:", total_loss / len(test_loader))
