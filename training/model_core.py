import os
import argparse
import json
import base64
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    resnet18, resnet50, convnext_tiny, convnext_base,
    ResNet18_Weights, ResNet50_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
)

from sklearn.metrics import (
    accuracy_score, roc_curve, auc, confusion_matrix,
    f1_score, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedShuffleSplit

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import warnings
from copy import deepcopy

# ===================== Global Config =====================
SEED = 27
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR = "train_test_outputs"

if multiprocessing.current_process().name == "MainProcess":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

USE_AMP = True

# ----------------- Training knobs -----------------
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0
BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

VAL_PATIENT_FRAC = 0.20

# ===================== Independent branch configuration =====================
# Directory names are configurable via command-line options.
MODALITY_FOLDERS = {
    "lesion_pre": "lesion_pre_nat",
    "lesion_post": "lesion_post_nat",
    "sln_gray": "sln_grayscale",
    "sln_pceus": "sln_pceus",
    "sln_ivceus": "sln_ivceus",
}

# ===================== Utils =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_ema(values, window_size=3, alpha=0.3):
    if len(values) == 0:
        return []
    ema_values = []
    for i in range(min(window_size - 1, len(values))):
        ema_values.append(float(np.mean(values[: i + 1])))
    if len(values) >= window_size:
        ema_prev = float(np.mean(values[:window_size]))
        ema_values.append(ema_prev)
        for i in range(window_size, len(values)):
            ema_current = float(alpha * values[i] + (1 - alpha) * ema_prev)
            ema_values.append(ema_current)
            ema_prev = ema_current
    return ema_values

# ===================== Plotting (ENGLISH) =====================
def plot_curves(train_losses, val_losses, test_losses, train_aucs, val_aucs, test_aucs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_losses, linewidth=2, label="Val Loss")
    if test_losses is not None and len(test_losses) == len(train_losses):
        ax1.plot(epochs, test_losses, linewidth=2, label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_aucs, linewidth=2, label="Train AUC")
    ax2.plot(epochs, val_aucs, linewidth=2, label="Val AUC")
    if test_aucs is not None and len(test_aucs) == len(train_aucs):
        ax2.plot(epochs, test_aucs, linewidth=2, label="Test AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC Curves")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ===================== Dataset =====================
class BreastCancerMultimodalDataset(Dataset):
    def __init__(
        self,
        data_root,
        df,
        transform=None,
        label_column="label",
        branch="lesion",
        modality_folders=None,
    ):
        self.data_root = data_root
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_column = label_column
        self.branch = branch
        self.modality_folders = modality_folders or MODALITY_FOLDERS

        if "case_number" not in self.df.columns:
            raise ValueError(f"Missing 'case_number' in DataFrame. Columns: {self.df.columns.tolist()}")
        if self.label_column not in self.df.columns:
            raise ValueError(f"Missing label column '{self.label_column}'. Columns: {self.df.columns.tolist()}")

    def _blank_image(self, size=(224, 224)):
        blank = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        return Image.fromarray(blank)

    def __len__(self):
        return len(self.df)

    def _load_image(self, folder_path):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"[Missing modality folder]\n{folder_path}")

        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON found in {folder_path}")

        json_path = os.path.join(folder_path, json_files[0])
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_data = base64.b64decode(data["imageData"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        roi_points = None
        if isinstance(data.get("shapes"), list) and len(data["shapes"]) > 0:
            for shp in data["shapes"]:
                if isinstance(shp, dict) and shp.get("points"):
                    roi_points = shp["points"]
                    break
        if roi_points is None and data.get("points"):
            roi_points = data["points"]

        if roi_points:
            points = np.array(roi_points, dtype=np.int32)
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [points], (255, 255, 255))
            roi_image = cv2.bitwise_and(image, mask)
            x, y, w, h = cv2.boundingRect(points)
            roi_image = roi_image[y : y + h, x : x + w]
            if roi_image.size < 100:
                roi_image = image
        else:
            roi_image = image

        return Image.fromarray(roi_image)

    def __getitem__(self, idx):
        case_info = self.df.iloc[idx]
        case_id = str(case_info["case_number"]).strip()
        case_id = case_id.replace("CASE_", "case_").replace("Case_", "case_")
        case_id = case_id.replace("CASE-", "case_").replace("Case-", "case_")
        if case_id.isdigit():
            case_id = f"case_{int(case_id):03d}"
        if case_id.lower().startswith("case_"):
            case_id = "case_" + case_id.split("_", 1)[1]

        label = case_info[self.label_column]

        case_path = os.path.join(self.data_root, case_id)
        pre_folder = os.path.join(case_path, self.modality_folders["lesion_pre"])
        post_folder = os.path.join(case_path, self.modality_folders["lesion_post"])
        modal1_folder = os.path.join(case_path, self.modality_folders["sln_gray"])
        modal2_folder = os.path.join(case_path, self.modality_folders["sln_pceus"])
        modal3_folder = os.path.join(case_path, self.modality_folders["sln_ivceus"])

        if self.branch == "lesion":
            images = (self._load_image(pre_folder), self._load_image(post_folder))
        elif self.branch == "sln":
            images = (self._load_image(modal1_folder), self._load_image(modal2_folder),
                      self._load_image(modal3_folder))
        else:
            raise ValueError(f"Unknown branch: {self.branch}")
        if self.transform:
            images = tuple(self.transform(img) for img in images)
        return (
            images,
            torch.tensor([label], dtype=torch.float32),
            case_info["number"],
            case_id,
        )


# ===================== Independent models =====================
class TemporalAttentionModule(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, feat_pre, feat_post):
        temporal_feats = torch.stack([feat_pre, feat_post], dim=1)
        avg_feat = temporal_feats.mean(dim=1)
        attn_weights = self.attention(avg_feat)
        return (temporal_feats * attn_weights.unsqueeze(2)).sum(dim=1)


def create_feature_extractor(model_name):
    if model_name == "resnet18":
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
    elif model_name == "resnet50":
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = base.fc.in_features
    elif model_name == "convnext_tiny":
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = base.classifier[2].in_features
    elif model_name == "convnext_base":
        base = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        in_features = base.classifier[2].in_features
    else:
        raise ValueError("Unsupported backbone")
    return nn.Sequential(*list(base.children())[:-1]), in_features


class BreastLesionModel(nn.Module):
    """Shared backbone for pre-/post-NAT grayscale images and temporal attention."""
    def __init__(self, model_name="resnet18", num_classes=1):
        super().__init__()
        self.extractor, feat_dim = create_feature_extractor(model_name)
        self.temporal_attn = TemporalAttentionModule(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        pre_img, post_img = x
        feat_pre = self.extractor(pre_img).flatten(1)
        feat_post = self.extractor(post_img).flatten(1)
        return self.classifier(self.temporal_attn(feat_pre, feat_post))


class SLNModel(nn.Module):
    """Three modality-specific backbones and four-head self-attention."""
    def __init__(self, model_name="resnet18", num_classes=1):
        super().__init__()
        self.gray_extractor, feat_dim = create_feature_extractor(model_name)
        self.pceus_extractor, _ = create_feature_extractor(model_name)
        self.ivceus_extractor, _ = create_feature_extractor(model_name)
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=4, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        gray, pceus, ivceus = x
        features = torch.stack([
            self.gray_extractor(gray).flatten(1),
            self.pceus_extractor(pceus).flatten(1),
            self.ivceus_extractor(ivceus).flatten(1),
        ], dim=1)
        attended, _ = self.cross_modal_attn(features, features, features)
        return self.classifier(attended.mean(dim=1))


# ===================== Metrics =====================
def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
    except Exception:
        fpr, tpr = [], []
        auc_score = 0.0

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ap_score = average_precision_score(y_true, y_prob)
    except Exception:
        precision, recall = [], []
        pr_auc = 0.0
        ap_score = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        if y_true[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, cm[0, 0], 0
    else:
        tn, fp, fn, tp = cm.ravel()

    sen = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "acc": float(acc),
        "auc": float(auc_score),
        "pr_auc": float(pr_auc),
        "ap_score": float(ap_score),
        "sen": float(sen),
        "spe": float(spe),
        "f1": float(f1),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
    }

def compute_pos_weight_from_df(df, label_col="label"):
    labels = df[label_col].values.astype(int)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0:
        return None
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)

@torch.no_grad()
def eval_loader(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    probs, preds, labels = [], [], []
    numbers, case_ids = [], []

    for images, y, nums, cids in loader:
        images = tuple(img.to(device) for img in images)
        y = y.to(device).squeeze(1)

        logits = model(images).squeeze(1)
        loss = criterion(logits, y)

        loss_sum += loss.item() * images[0].size(0)

        p = torch.sigmoid(logits).detach().cpu().numpy()
        pr = (p > 0.5).astype(int)

        probs.extend(p.tolist())
        preds.extend(pr.tolist())
        labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        numbers.extend(list(nums))
        case_ids.extend(list(cids))

    epoch_loss = loss_sum / len(loader.dataset)
    metrics = calculate_metrics(np.array(labels), np.array(preds), np.array(probs))
    metrics["loss"] = float(epoch_loss)
    return metrics, probs, preds, labels, numbers, case_ids

def train_model_with_val_and_test(model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, num_epochs, device, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    best_val_auc_ema = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False

    train_losses, val_losses, test_losses = [], [], []
    train_aucs, val_aucs, test_aucs = [], [], []
    val_auc_emas = []

    for epoch in range(num_epochs):
        # -------- train --------
        model.train()
        loss_sum = 0.0
        train_probs, train_preds, train_labels = [], [], []

        for images, y, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images = tuple(img.to(device) for img in images)
            y = y.to(device).squeeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * images[0].size(0)

            p = torch.sigmoid(logits).detach().cpu().numpy()
            pr = (p > 0.5).astype(int)
            train_probs.extend(p.tolist())
            train_preds.extend(pr.tolist())
            train_labels.extend(y.detach().cpu().numpy().astype(int).tolist())

        if scheduler is not None:
            scheduler.step()

        train_loss = loss_sum / len(train_loader.dataset)
        train_metrics = calculate_metrics(np.array(train_labels), np.array(train_preds), np.array(train_probs))
        train_metrics["loss"] = float(train_loss)

        # -------- val/test (no model selection on test) --------
        val_metrics, *_ = eval_loader(model, val_loader, criterion, device)
        test_metrics, *_ = eval_loader(model, test_loader, criterion, device) if test_loader is not None else (None,)*6

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        test_losses.append(test_metrics["loss"] if test_metrics else np.nan)

        train_aucs.append(train_metrics["auc"])
        val_aucs.append(val_metrics["auc"])
        test_aucs.append(test_metrics["auc"] if test_metrics else np.nan)

        val_auc_ema = calculate_ema(val_aucs)[-1]
        val_auc_emas.append(val_auc_ema)

        # select best ONLY by val EMA AUC
        if np.isfinite(val_auc_ema) and val_auc_ema > best_val_auc_ema + min_delta:
            epochs_without_improvement = 0
            best_val_auc_ema = val_auc_ema
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            epochs_without_improvement += 1

        print(f"[Epoch {epoch+1}] Train AUC={train_metrics['auc']:.4f} | Val AUC={val_metrics['auc']:.4f} (EMA={val_auc_ema:.4f}) | Test AUC={test_metrics['auc'] if test_metrics else float('nan'):.4f}")

        print(f"[EarlyStopping] No improvement: {epochs_without_improvement}/{patience} | Best epoch: {best_epoch}")
        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            print(f"[EarlyStopping] Stopped at epoch {epoch + 1}; restoring best epoch {best_epoch}.")
            break

    return {
        "epochs_ran": len(train_losses),
        "stopped_early": stopped_early,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "train_aucs": train_aucs,
        "val_aucs": val_aucs,
        "test_aucs": test_aucs,
        "val_auc_emas": val_auc_emas,
    }

def stratified_group_train_val_split(df, group_col="number", label_col="label", val_frac=0.2, seed=27):
    pt = df[[group_col, label_col]].drop_duplicates(subset=[group_col]).copy()
    pt[group_col] = pt[group_col].astype(str)
    y = pt[label_col].astype(int).values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(pt, y))

    tr_patients = set(pt.iloc[tr_idx][group_col].astype(str))
    va_patients = set(pt.iloc[va_idx][group_col].astype(str))

    train_sub = df[df[group_col].astype(str).isin(tr_patients)].copy()
    val = df[df[group_col].astype(str).isin(va_patients)].copy()
    return train_sub, val

def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# ===================== Main =====================
def main(branch=None):
    parser = argparse.ArgumentParser(description="Train an independent ultrasound imaging model")
    parser.add_argument("--excel-path", required=True, help="Local metadata spreadsheet; never commit patient data")
    parser.add_argument("--image-root", required=True, help="Local case directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience; 0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=EARLY_STOPPING_MIN_DELTA, help="Minimum improvement in validation AUC EMA")
    parser.add_argument("--group-column", default="number", help="Patient grouping field; verify patient uniqueness")
    for key, default_folder in MODALITY_FOLDERS.items():
        parser.add_argument("--" + key.replace("_", "-") + "-folder", default=default_folder)
    args = parser.parse_args()
    if branch not in ("lesion", "sln"):
        raise ValueError("branch must be lesion or sln")
    set_seed(SEED)
    EXCEL_PATH = args.excel_path
    IMAGE_ROOT = args.image_root
    global OUT_DIR
    OUT_DIR = args.output_dir or os.path.join("train_test_outputs", branch)
    for folder in (OUT_DIR, os.path.join(OUT_DIR, "best_models"),
                   os.path.join(OUT_DIR, "plots"), os.path.join(OUT_DIR, "predictions")):
        os.makedirs(folder, exist_ok=True)
    modality_folders = {key: getattr(args, key + "_folder") for key in MODALITY_FOLDERS}
    LABEL_COLUMN = "label"
    DATASET_COL = "dataset"  # train/test

    df = pd.read_excel(EXCEL_PATH)
    print(f"Total samples (rows): {len(df)}")
    if DATASET_COL not in df.columns:
        raise ValueError(f"Missing column '{DATASET_COL}' in Excel.")

    # standardize IDs
    df["case_number"] = df["case_number"].astype(str).str.strip()
    df["number"] = df["number"].astype(str).str.strip()
    df[args.group_column] = df[args.group_column].astype(str).str.strip()
    df[DATASET_COL] = df[DATASET_COL].astype(str).str.lower()

    # Optional deduplication by case_number (verify one record per SLN before use)
    dup_count = df["case_number"].duplicated().sum()
    if dup_count > 0:
        print(f"Found {dup_count} duplicated case_number. Dropping duplicates (keep first).")
        df = df.drop_duplicates(subset=["case_number"], keep="first")
        print(f"After dedup: {len(df)}")

    # split by dataset column
    df_train_all = df[df[DATASET_COL] == "train"].copy()
    df_test = df[df[DATASET_COL] == "test"].copy()
    if len(df_train_all) == 0 or len(df_test) == 0:
        raise ValueError("Empty train/test after split. Check dataset column values (train/test).")

    # train internal val split (patient-level, stratified)
    train_sub_df, val_df = stratified_group_train_val_split(
        df_train_all, group_col=args.group_column, label_col=LABEL_COLUMN, val_frac=VAL_PATIENT_FRAC, seed=SEED
    )

    print(f"[Split] Train(all)={len(df_train_all)} | Train(sub)={len(train_sub_df)} | Val={len(val_df)} | Test={len(df_test)}")
    print(f"[Patients] Train(all)={df_train_all[args.group_column].nunique()} | Train(sub)={train_sub_df[args.group_column].nunique()} | Val={val_df[args.group_column].nunique()} | Test={df_test[args.group_column].nunique()}")

    # transforms
    train_transform = get_transforms("train")
    eval_transform = get_transforms("eval")

    # datasets
    train_sub_dataset = BreastCancerMultimodalDataset(
        data_root=IMAGE_ROOT, df=train_sub_df, transform=train_transform,
        label_column=LABEL_COLUMN,
        branch=branch, modality_folders=modality_folders,
    )
    val_dataset = BreastCancerMultimodalDataset(
        data_root=IMAGE_ROOT, df=val_df, transform=eval_transform,
        label_column=LABEL_COLUMN,
        branch=branch, modality_folders=modality_folders,
    )
    test_dataset = BreastCancerMultimodalDataset(
        data_root=IMAGE_ROOT, df=df_test, transform=eval_transform,
        label_column=LABEL_COLUMN,
        branch=branch, modality_folders=modality_folders,
    )
    train_all_dataset = BreastCancerMultimodalDataset(
        data_root=IMAGE_ROOT, df=df_train_all, transform=eval_transform,
        label_column=LABEL_COLUMN,
        branch=branch, modality_folders=modality_folders,
    )

    # loaders
    train_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    train_all_loader = DataLoader(train_all_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # model
    model = BreastLesionModel() if branch == "lesion" else SLNModel()

    # loss pos_weight computed from train_sub only
    pos_weight = compute_pos_weight_from_df(train_sub_df, label_col=LABEL_COLUMN)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        print(f"[Loss] Using pos_weight = {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # train with internal val, test only for monitoring (NOT selection)
    hist = train_model_with_val_and_test(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=NUM_EPOCHS,
        device=device,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    # save best model
    best_path = os.path.join(OUT_DIR, "best_models", f"best_model_{branch}.pth")
    torch.save(hist["best_state"], best_path)
    print(f"[Saved] Best model: {best_path} (best epoch={hist['best_epoch']})")

    # load best and do final eval (train_all + val + test)
    model.load_state_dict(hist["best_state"])
    train_all_metrics, train_all_probs, train_all_preds, train_all_labels, train_all_numbers, train_all_case_ids = eval_loader(
        model, train_all_loader, criterion, device
    )
    val_metrics, val_probs, val_preds, val_labels, val_numbers, val_case_ids = eval_loader(
        model, val_loader, criterion, device
    )
    test_metrics, test_probs, test_preds, test_labels, test_numbers, test_case_ids = eval_loader(
        model, test_loader, criterion, device
    )

    # save predictions
    def save_pred_csv(name, probs, preds, labels, numbers, case_ids):
        pd.DataFrame({
            "number": numbers,
            "case_number": case_ids,
            "y_true": labels,
            "y_pred": preds,
            "y_prob": probs
        }).to_csv(os.path.join(OUT_DIR, "predictions", f"{name}.csv"), index=False, encoding="utf-8-sig")

    save_pred_csv("train_all_best_epoch", train_all_probs, train_all_preds, train_all_labels, train_all_numbers, train_all_case_ids)
    save_pred_csv("val_best_epoch", val_probs, val_preds, val_labels, val_numbers, val_case_ids)
    save_pred_csv("test_best_epoch", test_probs, test_preds, test_labels, test_numbers, test_case_ids)

    # plot curves
    plot_curves(
        hist["train_losses"], hist["val_losses"], hist["test_losses"],
        hist["train_aucs"], hist["val_aucs"], hist["test_aucs"],
        save_path=os.path.join(OUT_DIR, "plots", f"curves_{branch}.png")
    )

    # summary
    summary = {
        "branch": branch,
        "modality_folders": modality_folders,
        "num_epochs": NUM_EPOCHS,
        "epochs_ran": hist["epochs_ran"],
        "stopped_early": hist["stopped_early"],
        "early_stopping_patience": hist["early_stopping_patience"],
        "early_stopping_min_delta": hist["early_stopping_min_delta"],
        "best_epoch_by_val_auc_ema": hist["best_epoch"],
        "split_sizes": {
            "train_all_rows": int(len(df_train_all)),
            "train_sub_rows": int(len(train_sub_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(df_test)),
            "train_all_patients": int(df_train_all["number"].nunique()),
            "train_sub_patients": int(train_sub_df["number"].nunique()),
            "val_patients": int(val_df["number"].nunique()),
            "test_patients": int(df_test["number"].nunique()),
        },
        "best_epoch_metrics": {
            "train_all": train_all_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "curves": {
            "train_losses": hist["train_losses"],
            "val_losses": hist["val_losses"],
            "test_losses": hist["test_losses"],
            "train_aucs": hist["train_aucs"],
            "val_aucs": hist["val_aucs"],
            "test_aucs": hist["test_aucs"],
            "val_auc_emas": hist["val_auc_emas"],
        }
    }
    with open(os.path.join(OUT_DIR, f"summary_{branch}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Final (best epoch) ====")
    print("[Train-All]", {k: train_all_metrics[k] for k in ["auc", "pr_auc", "acc", "sen", "spe", "f1", "loss"]})
    print("[Val]",       {k: val_metrics[k] for k in ["auc", "pr_auc", "acc", "sen", "spe", "f1", "loss"]})
    print("[Test]",      {k: test_metrics[k] for k in ["auc", "pr_auc", "acc", "sen", "spe", "f1", "loss"]})
    print(f"\nSaved to: {OUT_DIR}/")