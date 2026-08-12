"""U-Net trainer for semantic segmentation."""
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
from tqdm import tqdm
from training.mlflow_manager import MLflowManager


class WeightedDiceLoss(torch.nn.Module):
    """Dice loss with per-class weights.

    SMP's DiceLoss has no ``class_weights`` argument, so this reimplements
    it: the one-hot masks are weighted per class before computing the Dice
    coefficient. Background covers ~80% of pixels, so unweighted Dice lets
    it dominate the gradient; these weights amplify the rare damage classes.
    """

    def __init__(self, weights, smooth=1.0):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        self.smooth = smooth

    def forward(self, logits, masks):
        """Compute weighted Dice loss."""
        probs = F.softmax(logits, dim=1)
        w = self.class_weights.view(1, -1, 1, 1)
        one_hot = torch.zeros_like(probs).scatter_(1, masks.unsqueeze(1), 1.0)
        intersection = (probs * one_hot * w).sum(dim=(2, 3)).sum(dim=0)
        cardinality = ((probs + one_hot) * w).sum(dim=(2, 3)).sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class UNetTrainer:
    """Trainer for U-Net models."""

    def __init__(self, config, mlflow_mgr: MLflowManager):
        """Initialize trainer."""
        self.config = config
        self.mlflow = mlflow_mgr
        self.model = None
        self.device = None
        self.best_loss = float('inf')

    def train(self, train_loader, valid_loader):
        """Train U-Net model."""
        print("Training U-Net...")
        self._seed_everything()
        self._setup_training()
        self._log_hyperparams()
        self._run_training_loop(train_loader, valid_loader)
        return self._get_results()

    def _seed_everything(self):
        """Fix RNG seeds so training is reproducible."""
        seed = getattr(self.config, "SEED", 2026)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ Seeded everything with SEED={seed}")

    def _setup_training(self):
        """Setup model and device."""
        self._create_model()
        self._setup_device()
        self._setup_criterion()
        self._setup_optimizer()

    def _create_model(self):
        """Create U-Net model."""
        self.model = smp.Unet(
            encoder_name=self.config.ENCODER,
            encoder_weights=self.config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=self.config.NUM_CLASSES
        )

    def _setup_device(self):
        """Setup device for training."""
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self.model.to(self.device)

    def _setup_criterion(self):
        """Setup loss function."""
        weights = getattr(self.config, "CLASS_WEIGHTS", None)
        if weights:
            self.criterion = WeightedDiceLoss(weights)
        else:
            self.criterion = smp.losses.DiceLoss(mode='multiclass')

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        lr = self.config.LEARNING_RATE
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=5
        )

    def _log_hyperparams(self):
        """Log hyperparameters."""
        params = self._get_hyperparams()
        self.mlflow.log_params(params)

    def _get_hyperparams(self):
        """Get hyperparameters dict."""
        return {
            "model_architecture": "unet",
            "encoder": self.config.ENCODER,
            "encoder_weights": self.config.ENCODER_WEIGHTS,
            "epochs": self.config.EPOCHS,
            "batch_size": self.config.BATCH_SIZE,
            "learning_rate": self.config.LEARNING_RATE,
            "num_classes": self.config.NUM_CLASSES,
            "img_size": self.config.IMG_SIZE,
            "device": str(self.device),
            "seed": getattr(self.config, "SEED", 2026),
        }

    def _run_training_loop(self, train_loader, valid_loader):
        """Run training loop."""
        patience_counter = 0
        for epoch in range(self.config.EPOCHS):
            if self._should_stop(patience_counter):
                break
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate_epoch(valid_loader, epoch)
            self._update_scheduler(val_loss)
            self._log_epoch_metrics(epoch, train_loss, val_loss)
            patience_counter = self._update_best(val_loss, patience_counter)

    def _train_epoch(self, loader, epoch):
        """Train single epoch."""
        self.model.train()
        total_loss = 0.0
        desc = f"Epoch {epoch+1}/{self.config.EPOCHS} [Train]"
        for images, masks in tqdm(loader, desc=desc):
            loss = self._train_batch(images, masks)
            total_loss += loss
        return total_loss / len(loader)

    def _train_batch(self, images, masks):
        """Train single batch."""
        images = images.to(self.device)
        masks = masks.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validate_epoch(self, loader, epoch):
        """Validate single epoch."""
        self.model.eval()
        total_loss = 0.0
        desc = f"Epoch {epoch+1}/{self.config.EPOCHS} [Valid]"
        with torch.no_grad():
            for images, masks in tqdm(loader, desc=desc):
                loss = self._validate_batch(images, masks)
                total_loss += loss
        return total_loss / len(loader)

    def _validate_batch(self, images, masks):
        """Validate single batch."""
        images = images.to(self.device)
        masks = masks.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        return loss.item()

    def _update_scheduler(self, val_loss):
        """Update learning rate scheduler."""
        self.scheduler.step(val_loss)

    def _log_epoch_metrics(self, epoch, train_loss, val_loss):
        """Log metrics for epoch."""
        self.mlflow.log_metric("train_loss", train_loss, epoch)
        self.mlflow.log_metric("val_loss", val_loss, epoch)
        lr = self.optimizer.param_groups[0]['lr']
        self.mlflow.log_metric("learning_rate", lr, epoch)
        msg = f"Epoch {epoch+1}/{self.config.EPOCHS}"
        print(f"{msg} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    def _update_best(self, val_loss, counter):
        """Update best model if improved."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_best_model(val_loss)
            return 0
        return counter + 1

    def _save_best_model(self, val_loss):
        """Save best model checkpoint."""
        path = "/opt/airflow/runs/semantic/unet_best.pth"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✓ Best model saved (val_loss: {val_loss:.4f})")

    def _should_stop(self, counter):
        """Check if should early stop."""
        if counter >= self.config.PATIENCE:
            print(f"Early stopping at patience {counter}")
            return True
        return False

    def _get_results(self):
        """Get training results."""
        self.mlflow.log_metric("best_val_loss", self.best_loss)
        return {"best_val_loss": self.best_loss}

    def save_model(self):
        """Save model to MLflow."""
        print("Saving model to MLflow...")
        self.mlflow.log_model(self.model, "model")
