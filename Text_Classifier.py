import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.nn.functional import softmax

def model_creation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=16)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    return model, tokenizer, train_loader, val_loader, optimizer, loss_fn, device

def model_trainer(model, tokenizer, train_loader, val_loader, optimizer, loss_fn, device):
    print("Entered model_trainer()")
    
    print("Starting training loop...")
    for epoch in range(3):
        print(f"Epoch {epoch+1} starting...")
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    print("Training done. Starting evaluation...")

    model.eval()
    predictions = []
    true_labels = []
    texts = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        texts.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "predicted_label": predictions,
        "correct": [int(t == p) for t, p in zip(true_labels, predictions)]
    })

    df.to_csv("rotten_tomatoes_predictions.csv", index=False)
    print("Predictions saved to rotten_tomatoes_predictions.csv")

def main():
    model, tokenizer, train_loader, val_loader, optimizer, loss_fn, device = model_creation()
    model_trainer(model, tokenizer, train_loader, val_loader, optimizer, loss_fn, device)

if __name__ == "__main__":
    main()
