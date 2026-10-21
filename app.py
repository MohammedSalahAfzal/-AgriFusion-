"""Flask demo for Agrifusion Net. Educational decision support only."""
import json, os
from pathlib import Path
from io import BytesIO
import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from torchvision import transforms
from model import AgrifusionNet

BASE = Path(__file__).resolve().parent
CHECKPOINT = Path(os.environ.get('AGRIFUSION_CHECKPOINT', BASE / 'artifacts/agrifusion_net.pt'))
ADVISORY_PATH = BASE / 'advisory.json'
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
CLASSES = []
IMAGE_SIZE = 224
ADVISORIES = json.loads(ADVISORY_PATH.read_text(encoding='utf-8')) if ADVISORY_PATH.exists() else {}


def load_model():
    global MODEL, CLASSES, IMAGE_SIZE
    if MODEL is not None: return
    if not CHECKPOINT.exists(): return
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    CLASSES = ckpt['classes']; IMAGE_SIZE = int(ckpt.get('image_size', 224))
    MODEL = AgrifusionNet(len(CLASSES), embed_dim=ckpt.get('embed_dim',256),
        num_heads=ckpt.get('num_heads',8), num_transformer_layers=ckpt.get('num_transformer_layers',2),
        dropout=ckpt.get('dropout',.25)).to(DEVICE)
    MODEL.load_state_dict(ckpt['model_state_dict']); MODEL.eval()


def get_advisory(label):
    key = label.lower().replace('_', ' ').replace('-', ' ')
    for name, info in ADVISORIES.items():
        if name.lower() in key or key in name.lower(): return info
    for token, info in ADVISORIES.items():
        if token.lower() in key: return info
    return {'summary':'No class-specific guidance is configured for this label.',
            'actions':['Retake a clear photo in natural light and inspect several leaves.','Ask a local agricultural extension officer or plant pathologist to confirm the cause.'],
            'caution':'The model output is a screening aid, not a confirmed diagnosis.'}


@app.route('/', methods=['GET','POST'])
def index():
    result = None; error = None
    if request.method == 'POST':
        if MODEL is None:
            error = 'No trained checkpoint found. Train the model first using the README instructions.'
        elif 'image' not in request.files or not request.files['image'].filename:
            error = 'Choose a leaf image to analyze.'
        else:
            file = request.files['image']
            try:
                image = Image.open(BytesIO(file.read())).convert('RGB')
                tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
                x = tf(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(MODEL(x), dim=1)[0].cpu()
                idx = int(torch.argmax(probs)); label = CLASSES[idx]
                result = {'label':label,'confidence':float(probs[idx]),'advisory':get_advisory(label),
                    'alternatives':sorted([{'label':CLASSES[i],'probability':float(probs[i])} for i in range(len(CLASSES))],key=lambda x:x['probability'],reverse=True)[:3]}
            except (UnidentifiedImageError, OSError): error = 'That file could not be read as an image. Please upload a JPG or PNG.'
            except Exception as exc:
                app.logger.exception('Prediction failed'); error = f'Prediction failed: {type(exc).__name__}. Check the terminal log.'
    return render_template('index.html', result=result, error=error, ready=MODEL is not None, classes=CLASSES)


@app.errorhandler(413)
def too_large(_):
    return render_template('index.html', result=None, error='Image too large. Maximum upload size is 10 MB.', ready=MODEL is not None, classes=CLASSES), 413

if __name__ == '__main__':
    load_model()
    app.run(host='127.0.0.1', port=5000, debug=False)
