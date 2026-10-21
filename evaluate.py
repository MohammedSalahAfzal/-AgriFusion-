"""Evaluate a saved checkpoint on a held-out test split."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from model import AgrifusionNet


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/rice_leaf_disease'); p.add_argument('--checkpoint',default='artifacts/agrifusion_net.pt'); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--workers',type=int,default=2); p.add_argument('--output',default='artifacts/evaluation.json'); args=p.parse_args()
    ckpt_path=Path(args.checkpoint)
    if not ckpt_path.exists(): raise SystemExit(f'Checkpoint not found: {ckpt_path}. Train first with train.py.')
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    classes=ckpt['classes']; size=ckpt.get('image_size',224); root=Path(args.data); test_dir=root/'test'
    if not test_dir.is_dir(): raise SystemExit(f'Missing {test_dir}. Create a held-out test split; see README.md.')
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(test_dir,transform=tf)
    if ds.classes != classes: raise SystemExit(f'Test classes {ds.classes} do not match checkpoint classes {classes}.')
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=max(0,args.workers))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=AgrifusionNet(len(classes),embed_dim=ckpt.get('embed_dim',256),num_heads=ckpt.get('num_heads',8),num_transformer_layers=ckpt.get('num_transformer_layers',2),dropout=ckpt.get('dropout',.25)).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1).cpu().numpy().tolist(); y_pred.extend(pred); y_true.extend(y.numpy().tolist())
    report=classification_report(y_true,y_pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    result={'checkpoint':str(ckpt_path),'test_samples':len(ds),'classes':classes,'accuracy':float(accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),'classification_report':report,'confusion_matrix':confusion_matrix(y_true,y_pred,labels=list(range(len(classes)))).tolist()}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print(f"Test samples: {len(ds)} | Accuracy: {result['accuracy']:.4f} | Macro F1: {result['macro_f1']:.4f}")
    print(classification_report(y_true,y_pred,target_names=classes,zero_division=0))
    print(f'Full report saved to {out}')

if __name__=='__main__': main()
