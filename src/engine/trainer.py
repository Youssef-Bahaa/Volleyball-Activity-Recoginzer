import torch
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score
from src.utils.logger import get_logger
import os

logger = get_logger('trainer', 'logs/app.log')


def run_epoch(model, loader, criterion, optimizer, num_classes, is_train, device='cpu'):
    """
    Runs a single train or val epoch.
    Returns (avg_loss, accuracy, macro_f1).
    """

    model.train() if is_train else model.eval()

    acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_metric  = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    total_loss, total = 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "  train" if is_train else "  val  "

    epoch_type = "Training" if is_train else "Validation"
    logger.debug(f"{epoch_type} epoch started on device: {device}")

    with ctx:
        for _, imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            preds = outputs.argmax(dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)


    avg_loss = total_loss / total
    accuracy = acc_metric.compute().item()
    macro_f1 = f1_metric.compute().item()

    logger.info(f"{epoch_type} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")


    return avg_loss, accuracy, macro_f1



def train(model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          device,
          num_classes,
          num_epochs,
          scheduler=None,
          checkpoint_dir='checkpoints',
          start_epoch=1,
          resume=None,
          save_best=True):
    """
    Full training loop using run_epoch for train/val.
    - model: nn.Module
    - train_loader / val_loader: DataLoader
    - criterion: loss
    - optimizer: optimizer
    - device: device string or torch.device
    - num_classes: int (for metrics)
    - num_epochs: total epochs to run
    - scheduler: optional LR scheduler (if ReduceLROnPlateau, pass 'scheduler' and will be stepped with val_loss)
    - checkpoint_dir: directory to save checkpoints
    - start_epoch: epoch to start from (useful when resuming)
    - resume: path to checkpoint to resume from (optional)
    - save_best: save best model by validation accuracy
    Returns (history dict)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = -1.0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}

    # resume if requested
    if resume:
        if os.path.exists(resume):
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt.get('model_state', ckpt))
            if 'optimizer_state' in ckpt and optimizer is not None:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            if scheduler is not None and 'scheduler_state' in ckpt and ckpt['scheduler_state'] is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                except Exception:
                    logger.debug("Could not load scheduler state, skipping.")
            start_epoch = ckpt.get('epoch', start_epoch)
            logger.info(f"Resumed from checkpoint {resume} starting at epoch {start_epoch}")
        else:
            logger.warning(f"Resume checkpoint not found: {resume}")

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs} started")
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer,
                                                    num_classes, is_train=True, device=device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        if val_loader is not None:
            val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer,
                                                  num_classes, is_train=False, device=device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
        else:
            val_loss, val_acc, val_f1 = None, None, None

        # scheduler step
        if scheduler is not None:
            # common pattern: ReduceLROnPlateau uses val_loss, others step per epoch
            if hasattr(scheduler, 'step') and scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                if val_loss is not None:
                    scheduler.step(val_loss)
            else:
                try:
                    scheduler.step()
                except Exception:
                    logger.debug("Scheduler step failed; skipping.")

        # save checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        }
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(ckpt, epoch_ckpt_path)
        logger.info(f"Saved checkpoint: {epoch_ckpt_path}")

        # save best
        if save_best and val_acc is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(checkpoint_dir, "best.pth")
                torch.save(ckpt, best_path)
                logger.info(f"New best model (val_acc={best_val_acc:.4f}) saved to {best_path}")

    logger.info("Training finished")
    return history

