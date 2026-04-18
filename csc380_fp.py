import numpy as np
import pandas as pd
import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# -----------------------
# Reproducibility, globals & settings
# -----------------------
SEED = 42
# Disable tokenizer warnings for clearer output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Loading & splitting dataset. Assumes dataset is in curr directory, and contains
# col split, so we can split the data with no test leakage. 
# -----------------------
df = pd.read_csv("edos_labelled_data.csv")
df = df[["text", "label", "split"]].dropna(subset=["text", "label"])
train_full_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

X_train_full_raw = train_full_df["text"].values
y_train_full = train_full_df["label"].values

X_test_raw = test_df["text"].values
y_test = test_df["label"].values

LABELS = ["sexist", "not sexist"]
label2id = {"not sexist": 0, "sexist": 1}
id2label = {0: "not sexist", 1: "sexist"}

# -----------------------
# Basic preprocessing for Random Forest + TF-IDF. We normalize the text by removing 
# url chars and leaving 'User' or 'url' placeholders and we lowercase the text, all
# to make text processing easier, and the model focus more on text patterns.
# -----------------------
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http[s]?://\S+", " URL ", text)
    text = re.sub(r"\[url\]", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\[user\]", " USER ", text)
    return text

# -----------------------
# Train/validation split (from TRAIN labeled data only
# in the dataset), so we can cross validate
# -----------------------
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
    X_train_full_raw, y_train_full, test_size=0.2,
    stratify=y_train_full, random_state=SEED, )

X_tr_clean = np.array([clean(t) for t in X_tr_raw])
X_val_clean = np.array([clean(t) for t in X_val_raw])
X_test_clean = np.array([clean(t) for t in X_test_raw])

# -----------------------
# Transformer dataset + model. We chose a model made in cardiff
# that apparently is good for text classifying tasks (Twitter-roBERTa-base).
#   - Model is pre-trained with offensive tweet data, suited for offensive
#   stuff dteection.
#   - We wrap the data into a custom dataset class (Ds), so we can iterate
#   through the data using PyTorch DataLoader (easier in my opinion)
#   - Tune the model (without an overkill, 1 epoch), to keep runtime reasonable
# -----------------------
class Ds(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        self.labels = [label2id[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

model_name = "cardiffnlp/twitter-roberta-base-offensive"    # Transformer used
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
train_dataset = Ds(X_tr_raw, y_tr, tokenizer)
val_dataset = Ds(X_val_raw, y_val, tokenizer)
test_dataset = Ds(X_test_raw, y_test, tokenizer)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Initially way more but first iteration gave the best results
num_epochs = 1

def eval_model_get_probs(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            sexist_probs = probs[:, 1]  # class 1 = "sexist"

            all_probs.append(sexist_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_probs, all_labels

print("Training Transformer Model (this might take a couple of minutes)...")
best_val_f1 = -1.0
best_state_dict = None

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    val_probs_tmp, val_labels_tmp = eval_model_get_probs(model, val_loader, device)
    val_preds_tmp = (val_probs_tmp >= 0.5).astype(int)
    _, _, val_f1_tmp, _ = precision_recall_fscore_support(
        val_labels_tmp,
        val_preds_tmp,
        average="weighted",
        zero_division=0,
    )
    if val_f1_tmp > best_val_f1:
        best_val_f1 = val_f1_tmp
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state_dict)
model.to(device)
val_probs, val_labels = eval_model_get_probs(model, val_loader,  device)
test_probs, test_labels = eval_model_get_probs(model, test_loader, device)

# -----------------------
# Ensemble of the model using Random Forest with char level TF-IDF
#   - We buikd a classical model with a 
# -----------------------
rf_vect = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    sublinear_tf=True,
)
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=SEED,
)

rf_pipe = Pipeline([
    ("tfidf", rf_vect),
    ("rf", rf_clf),
])

rf_pipe.fit(X_tr_clean, y_tr)

rf_val_proba = rf_pipe.predict_proba(X_val_clean)
rf_classes = rf_pipe.named_steps["rf"].classes_
idx_sexist_rf = list(rf_classes).index("sexist")
rf_val_probs = rf_val_proba[:, idx_sexist_rf]

rf_test_probs = rf_pipe.predict_proba(X_test_clean)[:, idx_sexist_rf]

# -----------------------
# Ensemble: tune threshold on VAL, evaluate on TEST
# -----------------------
ensemble_val_probs = 0.5 * rf_val_probs + 0.5 * val_probs

best_thr_ens = 0.5
best_f1_ens = -1.0

for thr in np.linspace(0.2, 0.8, 25):
    val_pred_ens = (ensemble_val_probs >= thr).astype(int)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        val_labels,
        val_pred_ens,
        average="weighted",
        zero_division=0,
    )
    if f1_w > best_f1_ens:
        best_f1_ens = f1_w
        best_thr_ens = thr

ensemble_test_probs = 0.5 * rf_test_probs + 0.5 * test_probs
test_pred_ens_int = (ensemble_test_probs >= best_thr_ens).astype(int)
test_pred_ens_str = np.array([id2label[i] for i in test_pred_ens_int])

report_ens = classification_report(
    y_test,
    test_pred_ens_str,
    labels=LABELS,
    target_names=LABELS,
    digits=4,
    zero_division=0,
)

print("\n=== Ensemble (RF-char-TFIDF + Twitter-RoBERTa, tuned thr) ===")
print(report_ens)