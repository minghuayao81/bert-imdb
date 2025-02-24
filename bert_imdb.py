import torch
from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification
)
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data.dataloader import default_collate
from pathlib import Path
from tqdm import tqdm

# Hypterparameters
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 5
ROOT_DIR = "/Users/minghuayao/AutoDL/"
OLD_MODEL_PATH = ROOT_DIR + "models/bert-base-uncased/"
NEW_MODEL_PATH = ROOT_DIR + "/models/new/"
DATA_SET = ROOT_DIR+ "aclImdb/"
NUM_LABELS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def load_aclimdb_directory(data_dir):
    data = []
    # traverse pos and neg folders
    # for label_type in ["pos", "neg", "unsup"]:
    for label_type in ["pos", "neg"]:
        dir_path = os.path.join(data_dir, label_type)
        if label_type == "pos":
            label = 0
        elif label_type == "neg":
            label = 1
        else:
            label = 2

        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as f:
                        text = f.read()
                        # dict: text and label
                        data.append({"text": text, "label": label})
                        
    return pd.DataFrame(data)

# load train set and test set
train_dir = os.path.join(DATA_SET, "train")
test_dir = os.path.join(DATA_SET, "test")
train_df = load_aclimdb_directory(train_dir)
test_df = load_aclimdb_directory(test_dir)

def load_and_preprocess_data():
    # load data
    train_df = load_aclimdb_directory(train_dir)
    test_df = load_aclimdb_directory(test_dir)

    tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL_PATH)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",  # auto padding to max length
            truncation=True,       
            max_length=MAX_LEN,    # set max length
            return_tensors="pt"    # return PyTorch tensor
        )
    
    # create dataset
    from datasets import Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # tokenizer
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # format tensor
    tokenized_train.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "label"]
    )
    tokenized_test.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "label"]
    )
    
    return DatasetDict({
        "train": tokenized_train.rename_column("label", "labels"),
        "test": tokenized_test.rename_column("label", "labels")
    })

def create_model():
    config = BertConfig.from_pretrained(
        OLD_MODEL_PATH,
        num_labels=NUM_LABELS
    )
    return BertForSequenceClassification.from_pretrained(OLD_MODEL_PATH, config=config)

def predict(new_model_path, text):
    try:
        # input validation
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text should not be empty")
            
        model_path = Path(new_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model does not exist：{model_path}")
            
        # Load model 
        # tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)

        # model
        model = BertForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=3,
            id2label={0: "POSITIVE", 1: "NEGATIVE", 2: "UNKNOWN"},
            label2id={"POSITIVE":0, "NEGATIVE":1, "UNKNOWN":2},
            local_files_only=True
        ).to(DEVICE)

        # preprocessing
        inputs = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        # prediction
        model.eval()
        with torch.inference_mode():
            outputs = model(**inputs)
        print(outputs.logits)

        # label casting
        label_map = getattr(model.config, "id2label", {})
        print(label_map)
        
        return label_map.get(outputs.logits.argmax().item(), "UNKNOWN")
        
    except Exception as e:
        error_msg = f"""
        Predict failed:
        1. input text length: {len(text)} characters
        2. error details: {str(e)}
        """        
        raise RuntimeError(error_msg) from e
        
def train_model(model, train_loader, val_loader):
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000
    )
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # grad
            optimizer.zero_grad()

            # fwd
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            # accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_train_correct += correct
            total_train_samples += labels.size(0)

            # backward and optim
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # train set average loss and accuracy
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_samples

        # evaluation
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                total_val_correct += correct
                total_val_samples += labels.size(0)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_val_correct / total_val_samples

        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

        # save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with val accuracy: {best_val_acc:.4f}')

    print('Training complete')
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

if __name__ == "__main__":    
    dataset = load_and_preprocess_data()
    tokenizer = BertTokenizer.from_pretrained(OLD_MODEL_PATH)
    
    model = create_model()

    train_loader = DataLoader(
        dataset["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset["test"],
        batch_size=32,
        collate_fn=collate_fn
    )

    train_model(model, train_loader, val_loader)

    def full_save_model(model, tokenizer, save_dir):
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    full_save_model(model, tokenizer, NEW_MODEL_PATH)

    
# 使用样例
print(predict(NEW_MODEL_PATH, "This movie is absolutely wonderful!"))

# 使用样例
print(predict(NEW_MODEL_PATH, "What a crap! Sucks!"))

# 使用样例
print(predict(NEW_MODEL_PATH, "love it"))