import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

def parse_text_file(file_path):
    descriptions = []
    criminals = []
    current_description = None
    current_criminal = None
    parsing_description = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace

            if line.startswith("**DESCRIPTION**"):
                # Start a new description
                current_description = ""
                parsing_description = True
                current_criminal = None

            elif line.startswith("**CRIMINAL**") and parsing_description:
                # Store the criminal name associated with the current description
                if current_description is not None:
                    criminal_name = line[12:].strip()
                    descriptions.append(current_description)
                    criminals.append(criminal_name)
                    parsing_description = False

            elif parsing_description:
                # Append to the current description
                if current_description is not None:
                    current_description += line + " "  # Add the line to the current description

    return descriptions, criminals

# Example usage:
file_path = "data.txt"  # Update with your file path
crime_descriptions, criminal_names = parse_text_file(file_path)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(criminal_names)

# Split data into train and test sets
train_descriptions, test_descriptions, train_labels, test_labels = train_test_split(
    crime_descriptions, encoded_labels, test_size=0.2, random_state=42
)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Prepare dataset class
class CrimeDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_length):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

assert len(criminal_names) == len(crime_descriptions), "Mismatch in labels and descriptions"
# Set up BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(encoded_labels)))

# Set training parameters
batch_size = 16
max_length = 256
epochs = 10
learning_rate = 5e-5

# Add data validation
print(f"Number of unique criminals: {len(set(criminal_names))}")
print(f"Total number of samples: {len(crime_descriptions)}")
print(f"Sample description: {crime_descriptions[0][:100]}...")  # Print first 100 chars of first description
print(f"Associated criminal: {criminal_names[0]}")

# Prepare train dataset and data loader
train_dataset = CrimeDataset(train_descriptions, train_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare test dataset and data loader
test_dataset = CrimeDataset(test_descriptions, test_labels, tokenizer, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss_list = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    loss_list.append(avg_loss)
    print(f'Train Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

    # Evaluate the model on test data
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f'Eval Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Test Epoch {epoch + 1}/{epochs} -- Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Save the trained model
model.save_pretrained('crime_bert_model')
tokenizer.save_pretrained('crime_bert_tokenizer')

print("Training complete. Model saved.")

import matplotlib.pyplot as plt

x = list(range(1, len(loss_list) + 1))  # One value per epoch
y = loss_list

plt.plot(x, y, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss Graph")
plt.grid(True)
plt.show()