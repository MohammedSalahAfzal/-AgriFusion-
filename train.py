"""Train Agrifusion Net from an ImageFolder dataset."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import AgrifusionNet


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/rice_leaf_disease', help='Root with train/val/test class folders')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pretrained', action='store_true', help='Download/use ImageNet pretrained EfficientNetV2-S weights')
    p.add_argument('--output', default='artifacts/agrifusion_net.pt')
    args = p.parse_args()
    seed_everything(args.seed)
    root = Path(args.data)
    required = ['train', 'val']
    for split in required:
        if not (root / split).is_dir():
            raise SystemExit(f'Missing {root / split}. See README.md for expected dataset layout.')
    size = args.image_size
    train_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12), transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_ds = datasets.ImageFolder(root / 'train', transform=train_tf)
    val_ds = datasets.ImageFolder(root / 'val', transform=eval_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f'Class folders differ. train={train_ds.classes}; val={val_ds.classes}')
    if len(train_ds.classes) < 2: raise SystemExit('Need at least two class folders.')
    workers = max(0, args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgrifusionNet(len(train_ds.classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=.5)
    best_acc = -1.0
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    print(f'Device: {device} | Classes: {train_ds.classes} | Train: {len(train_ds)} | Val: {len(val_ds)}')
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.; correct = 0; seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item(); seen += labels.size(0)
        train_loss = loss_sum / max(seen, 1); train_acc = correct / max(seen, 1)
        model.eval(); vloss_sum = 0.; vcorrect = 0; vseen = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images); loss = criterion(logits, labels)
                vloss_sum += loss.item() * labels.size(0)
                vcorrect += (logits.argmax(1) == labels).sum().item(); vseen += labels.size(0)
        val_loss = vloss_sum / max(vseen, 1); val_acc = vcorrect / max(vseen, 1)
        scheduler.step(val_acc)
        row = {'epoch':epoch,'train_loss':train_loss,'train_accuracy':train_acc,'val_loss':val_loss,'val_accuracy':val_acc}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict':model.state_dict(),'classes':train_ds.classes,
                'class_to_idx':train_ds.class_to_idx,'image_size':size,'architecture':'EfficientNetV2-S + Transformer Encoder',
                'embed_dim':256,'num_heads':8,'num_transformer_layers':2,'dropout':.25,
                'validation_accuracy_at_save':val_acc,'epoch':epoch}, out)
            print(f'  Saved best checkpoint: {out}')
    hist_path = out.with_name('training_history.json')
    hist_path.write_text(json.dumps(history, indent=2))
    print(f'Best validation accuracy: {best_acc:.4f} (fraction; multiply by 100 for percent)')
    print(f'History written to: {hist_path}')
    print('Next: run evaluate.py on a held-out test split. Validation accuracy is not test accuracy.')

if __name__ == '__main__': main()
