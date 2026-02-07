# DistilBERT: (A) multitask emotion+polarity vs (B) emotion-only, compare on same TEST(GPT98)
import os, json
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# -----------------------------
# CONFIG (edit paths)
# -----------------------------
TRAIN_FILE = r"C:\Users\Mustafa\Desktop\UNI\rcs\gpt_98\gpt_output_train.jsonl"
VALID_FILE = r"C:\Users\Mustafa\Desktop\UNI\rcs\gpt_98\gpt_output_validation.jsonl"
TEST_FILE  = r"C:\Users\Mustafa\Desktop\UNI\rcs\gpt_98\gpt_output_test.jsonl"

MODEL_NAME = "distilbert-base-uncased"  # Changed to DistilBERT
OUTPUT_ROOT = "./compare_emotion_polarity_outputs_distilbert"

MAX_LENGTH = 128
LR = 2e-5
EPOCHS = 6
BATCH = 32
SEED = 42

# loss weights for multitask
LAMBDA_EMO = 1.0
LAMBDA_POL = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
set_seed(SEED)

print("Device:", DEVICE)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# -----------------------------
# Helpers
# -----------------------------
def make_label_map(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {l: i for i, l in enumerate(uniq)}

def metrics_basic(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "f1_weighted": float(f1_w), "f1_macro": float(f1_m)}

def encode(tokenizer, texts, aspects, max_length=128):
    combined = [f"aspect: {a} text: {t}" for a, t in zip(aspects, texts)]
    return tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_token_type_ids=False,
    )

# -----------------------------
# Data loading (aspect-level) -> emotion+polarity
# Expects each aspect dict to have keys like: aspect, emotion, polarity
# -----------------------------
def load_jsonl_aspect_level_emotion_polarity(path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    texts, aspects, emotions, polarities = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "input" not in item or "output" not in item:
                continue

            review = item["input"]
            for a in item["output"]:
                if not isinstance(a, dict):
                    continue

                aspect   = str(a.get("aspect", "")).lower().strip()
                emotion  = str(a.get("emotion", "")).lower().strip()
                polarity = str(a.get("polarity", "")).lower().strip()

                # normalize common missing values
                def norm(x: str) -> str:
                    if x in ["", "masked"]:
                        return ""
                    if x == "null":
                        return "neutral"
                    return x

                emotion  = norm(emotion)
                polarity = norm(polarity)

                # for multitask we need BOTH labels present
                if emotion == "" or polarity == "":
                    continue

                texts.append(review)
                aspects.append(aspect)
                emotions.append(emotion)
                polarities.append(polarity)

    return texts, aspects, emotions, polarities

def load_jsonl_aspect_level_emotion_only(path: str) -> Tuple[List[str], List[str], List[str]]:
    texts, aspects, emotions = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "input" not in item or "output" not in item:
                continue

            review = item["input"]
            for a in item["output"]:
                if not isinstance(a, dict):
                    continue

                aspect  = str(a.get("aspect", "")).lower().strip()
                emotion = str(a.get("emotion", "")).lower().strip()

                if emotion in ["", "masked"]:
                    continue
                if emotion == "null":
                    emotion = "neutral"

                texts.append(review)
                aspects.append(aspect)
                emotions.append(emotion)

    return texts, aspects, emotions

# -----------------------------
# Dataset
# -----------------------------
class EncDataset(torch.utils.data.Dataset):
    def __init__(self, enc, y_em=None, y_pol=None, aspects=None):
        self.enc = enc
        self.y_em = y_em
        self.y_pol = y_pol
        self.aspects = aspects

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.enc["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.enc["attention_mask"][idx], dtype=torch.long),
        }
        if self.y_em is not None:
            item["labels_em"] = torch.tensor(self.y_em[idx], dtype=torch.long)
        if self.y_pol is not None:
            item["labels_pol"] = torch.tensor(self.y_pol[idx], dtype=torch.long)
        if self.aspects is not None:
            item["aspect_str"] = self.aspects[idx]
        return item

def collate(batch):
    out = {}
    for k in batch[0].keys():
        if k == "aspect_str":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out

# -----------------------------
# Models - DistilBERT versions
# -----------------------------
class DistilBertMultiTask(nn.Module):
    def __init__(self, model_name: str, n_em: int, n_pol: int, head_hidden: int = 128,
                 lambda_em: float = 1.0, lambda_pol: float = 1.0):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size

        self.head_em = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_em),
        )
        self.head_pol = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_pol),
        )

        self.lambda_em = lambda_em
        self.lambda_pol = lambda_pol
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels_em=None, labels_pol=None, **kwargs):
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", enabled=False):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT uses the first token ([CLS]) representation
        cls = out.last_hidden_state[:, 0, :]
        logits_em = self.head_em(cls)
        logits_pol = self.head_pol(cls)

        loss = None
        if labels_em is not None and labels_pol is not None:
            loss_em = self.ce(logits_em, labels_em)
            loss_pol = self.ce(logits_pol, labels_pol)
            loss = self.lambda_em * loss_em + self.lambda_pol * loss_pol

        return {"loss": loss, "logits_em": logits_em, "logits_pol": logits_pol}

class DistilBertEmotionOnly(nn.Module):
    def __init__(self, model_name: str, n_em: int, head_hidden: int = 128):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_em),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels_em=None, **kwargs):
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", enabled=False):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls)

        loss = None
        if labels_em is not None:
            loss = self.ce(logits, labels_em)

        return {"loss": loss, "logits_em": logits}

# -----------------------------
# Training args
# -----------------------------
def make_args(out_dir: str):
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=SEED,
    )

# -----------------------------
# Custom Trainer for multitask (so HuggingFace knows which tensors are labels)
# -----------------------------
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_em = inputs.pop("labels_em")
        labels_pol = inputs.pop("labels_pol")
        outputs = model(**inputs, labels_em=labels_em, labels_pol=labels_pol)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

class EmotionOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_em = inputs.pop("labels_em")
        outputs = model(**inputs, labels_em=labels_em)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# Reports
# -----------------------------
def print_report(title, y_true, y_pred, id2label):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("Summary:", metrics_basic(y_true, y_pred))
    labels_sorted = list(range(len(id2label)))
    names = [id2label[i] for i in labels_sorted]
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, labels=labels_sorted, target_names=names, zero_division=0))

# -----------------------------
# MAIN: load data
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)  # Changed to DistilBERT tokenizer

# (A) multitask data (needs emotion+polarity)
tr_t, tr_a, tr_e, tr_p = load_jsonl_aspect_level_emotion_polarity(TRAIN_FILE)
va_t, va_a, va_e, va_p = load_jsonl_aspect_level_emotion_polarity(VALID_FILE)
te_t, te_a, te_e, te_p = load_jsonl_aspect_level_emotion_polarity(TEST_FILE)

print("Multitask Train/Val/Test:", len(tr_t), len(va_t), len(te_t))

emotion2id = make_label_map(tr_e)
pol2id     = make_label_map(tr_p)
id2emotion = {v: k for k, v in emotion2id.items()}
id2pol     = {v: k for k, v in pol2id.items()}

print("Emotions:", emotion2id)
print("Polarities:", pol2id)

def filter_seen_multitask(texts, aspects, emotions, pols):
    keep = [(e in emotion2id) and (p in pol2id) for e, p in zip(emotions, pols)]
    return (
        [t for t, k in zip(texts, keep) if k],
        [a for a, k in zip(aspects, keep) if k],
        [e for e, k in zip(emotions, keep) if k],
        [p for p, k in zip(pols, keep) if k],
    )

va_t, va_a, va_e, va_p = filter_seen_multitask(va_t, va_a, va_e, va_p)
te_t, te_a, te_e, te_p = filter_seen_multitask(te_t, te_a, te_e, te_p)

ytr_e = np.array([emotion2id[x] for x in tr_e], dtype=np.int64)
yva_e = np.array([emotion2id[x] for x in va_e], dtype=np.int64)
yte_e = np.array([emotion2id[x] for x in te_e], dtype=np.int64)

ytr_p = np.array([pol2id[x] for x in tr_p], dtype=np.int64)
yva_p = np.array([pol2id[x] for x in va_p], dtype=np.int64)
yte_p = np.array([pol2id[x] for x in te_p], dtype=np.int64)

enc_tr = encode(tokenizer, tr_t, tr_a, MAX_LENGTH)
enc_va = encode(tokenizer, va_t, va_a, MAX_LENGTH)
enc_te = encode(tokenizer, te_t, te_a, MAX_LENGTH)

ds_tr_mt = EncDataset(enc_tr, y_em=ytr_e, y_pol=ytr_p, aspects=tr_a)
ds_va_mt = EncDataset(enc_va, y_em=yva_e, y_pol=yva_p, aspects=va_a)
ds_te_mt = EncDataset(enc_te, y_em=yte_e, y_pol=yte_p, aspects=te_a)

# (B) emotion-only data 
tr_t2, tr_a2, tr_e2 = load_jsonl_aspect_level_emotion_only(TRAIN_FILE)
va_t2, va_a2, va_e2 = load_jsonl_aspect_level_emotion_only(VALID_FILE)
te_t2, te_a2, te_e2 = load_jsonl_aspect_level_emotion_only(TEST_FILE)

# emotion map for emotion-only using its own train labels
emotion2id_eo = make_label_map(tr_e2)
id2emotion_eo = {v: k for k, v in emotion2id_eo.items()}
print("Emotion-only Emotions:", emotion2id_eo)

def filter_seen_emotion_only(texts, aspects, emotions):
    keep = [(e in emotion2id_eo) for e in emotions]
    return (
        [t for t, k in zip(texts, keep) if k],
        [a for a, k in zip(aspects, keep) if k],
        [e for e, k in zip(emotions, keep) if k],
    )

va_t2, va_a2, va_e2 = filter_seen_emotion_only(va_t2, va_a2, va_e2)

ytr_e2 = np.array([emotion2id_eo[x] for x in tr_e2], dtype=np.int64)
yva_e2 = np.array([emotion2id_eo[x] for x in va_e2], dtype=np.int64)

enc_tr2 = encode(tokenizer, tr_t2, tr_a2, MAX_LENGTH)
enc_va2 = encode(tokenizer, va_t2, va_a2, MAX_LENGTH)

ds_tr_eo = EncDataset(enc_tr2, y_em=ytr_e2, y_pol=None, aspects=tr_a2)
ds_va_eo = EncDataset(enc_va2, y_em=yva_e2, y_pol=None, aspects=va_a2)

print("\nRebuilding emotion-only model label space to match multitask emotion labels for fair comparison...")
emotion2id_fair = emotion2id
id2emotion_fair = id2emotion

def filter_to_fair_emotion_space(texts, aspects, emotions):
    keep = [(e in emotion2id_fair) for e in emotions]
    return (
        [t for t, k in zip(texts, keep) if k],
        [a for a, k in zip(aspects, keep) if k],
        [e for e, k in zip(emotions, keep) if k],
    )

tr_tF, tr_aF, tr_eF = filter_to_fair_emotion_space(tr_t2, tr_a2, tr_e2)
va_tF, va_aF, va_eF = filter_to_fair_emotion_space(va_t2, va_a2, va_e2)

ytr_eF = np.array([emotion2id_fair[x] for x in tr_eF], dtype=np.int64)
yva_eF = np.array([emotion2id_fair[x] for x in va_eF], dtype=np.int64)

enc_trF = encode(tokenizer, tr_tF, tr_aF, MAX_LENGTH)
enc_vaF = encode(tokenizer, va_tF, va_aF, MAX_LENGTH)

ds_tr_eo_fair = EncDataset(enc_trF, y_em=ytr_eF, y_pol=None, aspects=tr_aF)
ds_va_eo_fair = EncDataset(enc_vaF, y_em=yva_eF, y_pol=None, aspects=va_aF)

# TEST for emotion-only : use multitask test enc_te and labels yte_e
ds_te_eo_fair = EncDataset(enc_te, y_em=yte_e, y_pol=None, aspects=te_a)

# -----------------------------
# Train (A) MULTITASK
# -----------------------------
print("\n\n### (A) DISTILBERT MULTITASK MODEL (emotion + polarity) ###")
out_dir_mt = os.path.join(OUTPUT_ROOT, "distilbert_multitask_em_pol")
os.makedirs(out_dir_mt, exist_ok=True)

model_mt = DistilBertMultiTask(
    MODEL_NAME,
    n_em=len(emotion2id),
    n_pol=len(pol2id),
    head_hidden=128,
    lambda_em=LAMBDA_EMO,
    lambda_pol=LAMBDA_POL,
)

trainer_mt = MultiTaskTrainer(
    model=model_mt,
    args=make_args(out_dir_mt),
    train_dataset=ds_tr_mt,
    eval_dataset=ds_va_mt,
    data_collator=collate,
)

trainer_mt.train()
trainer_mt.save_model(out_dir_mt)
tokenizer.save_pretrained(out_dir_mt)

# Predict on test
pred_mt = trainer_mt.predict(ds_te_mt)
logits_em_mt = pred_mt.predictions[0] if isinstance(pred_mt.predictions, (tuple, list)) else pred_mt.predictions

if isinstance(pred_mt.predictions, dict):
    logits_em_mt = pred_mt.predictions["logits_em"]
    logits_pol_mt = pred_mt.predictions["logits_pol"]
elif isinstance(pred_mt.predictions, (tuple, list)) and len(pred_mt.predictions) == 2:
    logits_em_mt, logits_pol_mt = pred_mt.predictions
else:
    raise RuntimeError("Unexpected prediction format from multitask model.")

yhat_em_mt = np.argmax(logits_em_mt, axis=1)
yhat_pol_mt = np.argmax(logits_pol_mt, axis=1)

print_report("DISTILBERT MULTITASK on TEST — Emotion", yte_e, yhat_em_mt, id2emotion)
print_report("DISTILBERT MULTITASK on TEST — Polarity", yte_p, yhat_pol_mt, id2pol)

mt_em = metrics_basic(yte_e, yhat_em_mt)
mt_pol = metrics_basic(yte_p, yhat_pol_mt)

# -----------------------------
# Train (B) EMOTION-ONLY 
# -----------------------------
print("\n\n### (B) DISTILBERT EMOTION-ONLY MODEL (trained only on emotion) ###")
out_dir_eo = os.path.join(OUTPUT_ROOT, "distilbert_emotion_only_fair")
os.makedirs(out_dir_eo, exist_ok=True)

model_eo = DistilBertEmotionOnly(MODEL_NAME, n_em=len(emotion2id_fair), head_hidden=128)

trainer_eo = EmotionOnlyTrainer(
    model=model_eo,
    args=make_args(out_dir_eo),
    train_dataset=ds_tr_eo_fair,
    eval_dataset=ds_va_eo_fair,
    data_collator=collate,
)

trainer_eo.train()
trainer_eo.save_model(out_dir_eo)
tokenizer.save_pretrained(out_dir_eo)

pred_eo = trainer_eo.predict(ds_te_eo_fair)
logits_em_eo = pred_eo.predictions
yhat_em_eo = np.argmax(logits_em_eo, axis=1)

print_report("DISTILBERT EMOTION-ONLY on TEST — Emotion", yte_e, yhat_em_eo, id2emotion_fair)

eo_em = metrics_basic(yte_e, yhat_em_eo)

print("\n" + "#" * 70)
print("FINAL COMPARISON (same TEST set, same emotion label space)")
print("#" * 70)
print("DistilBERT Emotion-only   :", eo_em)
print("DistilBERT Multitask Emo  :", mt_em)
print("DistilBERT Multitask Pol  :", mt_pol)

print("\nDONE.")