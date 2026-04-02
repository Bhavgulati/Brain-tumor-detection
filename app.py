import flask
import sqlite3
import os
import io
import datetime
import base64
import json
import numpy as np
import cv2
from io import BytesIO
from functools import wraps

import torch
from torch import argmax, load, softmax
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import resnet50
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join('static', 'photos')
DB_PATH = 'patient_history.db'
DOCTOR_USERNAME = 'doctor'
DOCTOR_PASSWORD = 'doctor@123'   # Change this in production

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "brain_tumor_secret_2024"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ── Labels & Info ─────────────────────────────────────────────────────────────
BRAIN_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
BRAIN_INFO = {
    'No Tumor':   'No tumor was detected in the MRI scan. The brain tissue appears normal.',
    'Meningioma': 'A typically slow-growing tumor that forms on the membranes covering the brain and spinal cord. Usually benign and treatable.',
    'Glioma':     'A tumor that originates in the glial cells of the brain or spinal cord. Requires immediate medical attention.',
    'Pituitary':  'A tumor that forms in the pituitary gland at the base of the brain. Usually benign and often manageable.',
}
BRAIN_SEVERITY = {
    'No Tumor': 'normal', 'Meningioma': 'warning', 'Glioma': 'danger', 'Pituitary': 'warning'
}

SKIN_LABELS = ['Benign', 'Malignant']
SKIN_INFO = {
    'Benign':    'The lesion appears non-cancerous. Regular monitoring is still recommended.',
    'Malignant': 'The lesion shows signs of malignancy. Immediate dermatologist consultation is strongly advised.',
}
SKIN_SEVERITY = {
    'Benign': 'normal', 'Malignant': 'danger'
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute('''CREATE TABLE IF NOT EXISTS scans (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name    TEXT NOT NULL,
        patient_age     TEXT,
        patient_email   TEXT,
        scan_date       TEXT NOT NULL,
        scan_type       TEXT NOT NULL DEFAULT 'brain',
        result          TEXT NOT NULL,
        confidence      REAL NOT NULL,
        tumor_size_px   INTEGER,
        image_path      TEXT,
        gradcam_path    TEXT,
        all_confidences TEXT
    )''')

    conn.execute('''CREATE TABLE IF NOT EXISTS doctor_notes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id     INTEGER NOT NULL,
        note        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        FOREIGN KEY (scan_id) REFERENCES scans(id)
    )''')

    conn.execute('''CREATE TABLE IF NOT EXISTS patients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        age         TEXT,
        email       TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )''')

    conn.commit()
    conn.close()

init_db()

def save_scan(patient_name, patient_age, patient_email, scan_type, result,
              confidence, image_path, gradcam_path, tumor_size_px, all_confidences):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        '''INSERT INTO scans
           (patient_name,patient_age,patient_email,scan_date,scan_type,result,
            confidence,tumor_size_px,image_path,gradcam_path,all_confidences)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        (patient_name, patient_age, patient_email,
         datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
         scan_type, result, confidence, tumor_size_px,
         image_path, gradcam_path, json.dumps(all_confidences))
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_all_scans(scan_type=None):
    conn = sqlite3.connect(DB_PATH)
    if scan_type:
        rows = conn.execute('SELECT * FROM scans WHERE scan_type=? ORDER BY id DESC', (scan_type,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM scans ORDER BY id DESC').fetchall()
    conn.close()
    return rows

def get_scan_by_id(scan_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('SELECT * FROM scans WHERE id=?', (scan_id,)).fetchone()
    conn.close()
    return row

def get_patient_scans(patient_email):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT * FROM scans WHERE patient_email=? ORDER BY id DESC', (patient_email,)).fetchall()
    conn.close()
    return rows

def save_note(scan_id, note):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO doctor_notes (scan_id,note,created_at) VALUES (?,?,?)',
                 (scan_id, note, datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    conn.commit()
    conn.close()

def get_notes(scan_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT * FROM doctor_notes WHERE scan_id=? ORDER BY id DESC', (scan_id,)).fetchall()
    conn.close()
    return rows

def get_patient_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('SELECT * FROM patients WHERE email=?', (email,)).fetchone()
    conn.close()
    return row

def register_patient(name, age, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO patients (name,age,email,password,created_at) VALUES (?,?,?,?,?)',
                     (name, age, email, password, datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# ── Auth Decorators ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'patient_email' not in flask.session:
            return flask.redirect(flask.url_for('patient_login'))
        return f(*args, **kwargs)
    return decorated

def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not flask.session.get('is_doctor'):
            return flask.redirect(flask.url_for('doctor_login'))
        return f(*args, **kwargs)
    return decorated

# ── Models ────────────────────────────────────────────────────────────────────
device = "cuda" if is_available() else "cpu"

# Brain Tumor Model
brain_model = resnet50(weights=None)
n_inputs = brain_model.fc.in_features
brain_model.fc = Sequential(
    Linear(n_inputs, 2048), SELU(), Dropout(p=0.4),
    Linear(2048, 2048),     SELU(), Dropout(p=0.4),
    Linear(2048, 4)
)
brain_model.to(device)
if os.path.exists('./models/bt_resnet50_model.pt'):
    brain_model.load_state_dict(load('./models/bt_resnet50_model.pt', map_location=DEVICE(device)))
brain_model.eval()

# Skin Cancer Model (ResNet50, 2 classes — benign vs malignant)
skin_model = resnet50(weights=None)
n_inputs_skin = skin_model.fc.in_features
skin_model.fc = Sequential(
    Linear(n_inputs_skin, 512), SELU(), Dropout(p=0.4),
    Linear(512, 2)
)
skin_model.to(device)
if os.path.exists('./models/skin_resnet50_model.pt'):
    skin_model.load_state_dict(load('./models/skin_resnet50_model.pt', map_location=DEVICE(device)))
skin_model.eval()

# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_id):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_id].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

brain_gradcam = GradCAM(brain_model)
skin_gradcam  = GradCAM(skin_model)

def generate_gradcam_overlay(image_bytes, cam):
    """Return base64 PNG of original image side-by-side with GradCAM overlay."""
    orig    = Image.open(BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    orig_np = np.array(orig)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
    combined = np.concatenate([orig_np, overlay], axis=1)
    buf = BytesIO()
    Image.fromarray(combined).save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ── Tumor Size Estimation ─────────────────────────────────────────────────────
def estimate_tumor_size(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('L').resize((224, 224))
        arr = np.array(img)
        _, thresh = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return int(cv2.contourArea(max(contours, key=cv2.contourArea)))
        return 0
    except Exception:
        return 0

# ── Preprocessing & Prediction ────────────────────────────────────────────────
def preprocess_image(image_bytes):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)

def get_prediction(image_bytes, scan_type='brain'):
    model  = brain_model if scan_type == 'brain' else skin_model
    labels = BRAIN_LABELS if scan_type == 'brain' else SKIN_LABELS
    gcam   = brain_gradcam if scan_type == 'brain' else skin_gradcam

    tensor = preprocess_image(image_bytes).to(device)
    tensor.requires_grad_(True)

    y_hat     = model(tensor)
    probs     = softmax(y_hat, dim=1)[0]
    class_id  = int(argmax(probs))
    conf_list = [round(float(p) * 100, 2) for p in probs]

    cam         = gcam.generate(tensor, class_id)
    gradcam_b64 = generate_gradcam_overlay(image_bytes, cam)

    return class_id, labels[class_id], conf_list, gradcam_b64

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ── PDF Report ────────────────────────────────────────────────────────────────
def generate_pdf(patient_name, patient_age, scan_type, result, confidence,
                 all_confidences, labels, image_path, scan_date, notes=None):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch,   bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_s   = ParagraphStyle('T', parent=styles['Title'],  fontSize=20,
                                textColor=colors.HexColor('#1a365d'), spaceAfter=4, alignment=TA_CENTER)
    sub_s     = ParagraphStyle('S', parent=styles['Normal'], fontSize=11,
                                textColor=colors.HexColor('#4a5568'), spaceAfter=16, alignment=TA_CENTER)
    section_s = ParagraphStyle('H', parent=styles['Normal'], fontSize=13,
                                fontName='Helvetica-Bold', textColor=colors.HexColor('#2d3748'), spaceAfter=8)
    disc_s    = ParagraphStyle('D', parent=styles['Normal'], fontSize=9,
                                textColor=colors.HexColor('#718096'),
                                borderColor=colors.HexColor('#e2e8f0'), borderWidth=1,
                                borderPadding=8, backColor=colors.HexColor('#f7fafc'))

    scan_label = 'Brain Tumor' if scan_type == 'brain' else 'Skin Cancer'
    story.append(Paragraph(f"{scan_label} Detection Report", title_s))
    story.append(Paragraph("AI-Powered Medical Imaging Analysis — For Research Use Only", sub_s))
    story.append(Table([['']], colWidths=[6.5*inch],
                        style=TableStyle([('LINEABOVE',(0,0),(0,0),1.5,colors.HexColor('#3182ce'))])))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Patient Information", section_s))
    info_table = Table([
        ['Patient Name', patient_name or 'N/A'],
        ['Age',          patient_age  or 'N/A'],
        ['Scan Type',    scan_label],
        ['Scan Date',    scan_date],
        ['Report Date',  datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
    ], colWidths=[2*inch, 4.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#ebf8ff')),
        ('TEXTCOLOR', (0,0),(0,-1),colors.HexColor('#2b6cb0')),
        ('FONTNAME',  (0,0),(0,-1),'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1),11),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#f7fafc'),colors.white]),
        ('GRID',      (0,0),(-1,-1),0.5,colors.HexColor('#cbd5e0')),
        ('PADDING',   (0,0),(-1,-1),8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Diagnosis", section_s))
    label_info = BRAIN_INFO if scan_type == 'brain' else SKIN_INFO
    dc = '#276749' if result in ('No Tumor', 'Benign') else '#9b2c2c'
    diag_table = Table([
        ['Detected Condition', result],
        ['Confidence',         f'{confidence:.1f}%'],
        ['Description',        label_info.get(result, '')],
    ], colWidths=[2*inch, 4.5*inch])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#fef3c7')),
        ('TEXTCOLOR', (1,0),(1,0),colors.HexColor(dc)),
        ('FONTNAME',  (0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',  (1,0),(1,0),'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1),11),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#fffaf0'),colors.white]),
        ('GRID',      (0,0),(-1,-1),0.5,colors.HexColor('#cbd5e0')),
        ('PADDING',   (0,0),(-1,-1),8),
        ('VALIGN',    (0,0),(-1,-1),'TOP'),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Class Probabilities", section_s))
    prob_data = [['Condition', 'Confidence (%)']]
    for label, conf in zip(labels, all_confidences):
        prob_data.append([label, f'{conf:.2f}%'])
    prob_table = Table(prob_data, colWidths=[3.25*inch, 3.25*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0,0),(-1,0),colors.white),
        ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1),11),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#ebf8ff'),colors.white]),
        ('GRID',      (0,0),(-1,-1),0.5,colors.HexColor('#bee3f8')),
        ('ALIGN',     (1,0),(-1,-1),'CENTER'),
        ('PADDING',   (0,0),(-1,-1),8),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 16))

    if notes:
        story.append(Paragraph("Doctor Notes", section_s))
        for note in notes:
            story.append(Paragraph(f"[{note[3]}] {note[2]}", disc_s))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 10))

    if image_path and os.path.exists(image_path):
        story.append(Paragraph("Uploaded Scan", section_s))
        try:
            story.append(RLImage(image_path, width=3*inch, height=3*inch))
        except Exception:
            pass
        story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an AI system for research and educational "
        "purposes only. It is not a substitute for professional medical advice. "
        "Always consult a qualified healthcare provider.", disc_s))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Routes: Main ──────────────────────────────────────────────────────────────
@app.route('/')
def main():
    scans = get_all_scans()
    total        = len(scans)
    tumor_count  = sum(1 for s in scans if s[6] not in ('No Tumor', 'Benign'))
    brain_count  = sum(1 for s in scans if s[5] == 'brain')
    skin_count   = sum(1 for s in scans if s[5] == 'skin')
    return flask.render_template('DiseaseDet.html',
        total=total, tumor_count=tumor_count,
        brain_count=brain_count, skin_count=skin_count,
        recent=scans[:5])

# ── Routes: Upload & Predict ──────────────────────────────────────────────────
@app.route('/uimg', methods=['GET', 'POST'])
def uimg():
    if flask.request.method == 'GET':
        return flask.render_template('uimg.html')

    file          = flask.request.files.get('file')
    patient_name  = flask.request.form.get('patient_name', 'Unknown')
    patient_age   = flask.request.form.get('patient_age', 'N/A')
    patient_email = flask.request.form.get('patient_email', '')
    scan_type     = flask.request.form.get('scan_type', 'brain')  # 'brain' or 'skin'

    if not file or not allowed_file(file.filename):
        return flask.redirect(flask.url_for('uimg'))

    img_bytes = file.read()
    class_id, class_name, all_confidences, gradcam_b64 = get_prediction(img_bytes, scan_type)
    top_conf  = all_confidences[class_id]
    scan_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    # Save original image
    filename   = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    Image.open(BytesIO(img_bytes)).convert('RGB').save(image_path)

    # Save GradCAM image to disk
    gradcam_filename = f"gradcam_{filename}"
    gradcam_path     = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    with open(gradcam_path, 'wb') as f:
        f.write(base64.b64decode(gradcam_b64))

    # Tumor size (only when something is detected)
    tumor_size_px = estimate_tumor_size(img_bytes) if class_name not in ('No Tumor', 'Benign') else 0

    scan_id = save_scan(patient_name, patient_age, patient_email, scan_type,
                        class_name, top_conf, image_path, gradcam_path,
                        tumor_size_px, all_confidences)

    labels   = BRAIN_LABELS   if scan_type == 'brain' else SKIN_LABELS
    info_map = BRAIN_INFO     if scan_type == 'brain' else SKIN_INFO
    sev_map  = BRAIN_SEVERITY if scan_type == 'brain' else SKIN_SEVERITY

    return flask.render_template('pred.html',
        scan_id=scan_id,
        result=class_name,
        confidence=top_conf,
        all_confidences=all_confidences,
        labels=labels,
        description=info_map.get(class_name, ''),
        severity=sev_map.get(class_name, 'warning'),
        patient_name=patient_name,
        patient_age=patient_age,
        patient_email=patient_email,
        scan_type=scan_type,
        scan_date=scan_date,
        image_path=image_path,
        filename=filename,
        gradcam_b64=gradcam_b64,
        tumor_size_px=tumor_size_px,
    )

# ── Routes: History ───────────────────────────────────────────────────────────
@app.route('/history')
def history():
    scan_type = flask.request.args.get('type', None)
    scans = get_all_scans(scan_type)
    return flask.render_template('history.html', scans=scans, scan_type=scan_type)

# ── Routes: Scan Detail ───────────────────────────────────────────────────────
@app.route('/scan/<int:scan_id>')
def scan_detail(scan_id):
    scan  = get_scan_by_id(scan_id)
    notes = get_notes(scan_id)
    if not scan:
        return flask.abort(404)
    all_conf = json.loads(scan[11]) if scan[11] else []
    labels   = BRAIN_LABELS if scan[5] == 'brain' else SKIN_LABELS
    return flask.render_template('scan_detail.html',
        scan=scan, notes=notes, all_conf=all_conf, labels=labels,
        is_doctor=flask.session.get('is_doctor', False))

# ── Routes: Compare Scans ─────────────────────────────────────────────────────
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    scans  = get_all_scans()
    scan_a, scan_b = None, None
    if flask.request.method == 'POST':
        id_a = flask.request.form.get('scan_a')
        id_b = flask.request.form.get('scan_b')
        if id_a:
            scan_a = get_scan_by_id(int(id_a))
        if id_b:
            scan_b = get_scan_by_id(int(id_b))
    return flask.render_template('compare.html', scans=scans, scan_a=scan_a, scan_b=scan_b)

# ── Routes: Statistics Dashboard ─────────────────────────────────────────────
@app.route('/stats')
def stats():
    from collections import Counter, defaultdict
    scans = get_all_scans()
    total = len(scans)

    result_counts = dict(Counter(s[6] for s in scans))
    tumor_count   = sum(1 for s in scans if s[6] not in ('No Tumor', 'Benign'))
    normal_count  = total - tumor_count

    recent = scans[:20][::-1]
    trend_dates  = [s[4] for s in recent]
    trend_conf   = [round(s[7], 2) for s in recent]
    trend_labels = [s[6] for s in recent]

    conf_by_type = defaultdict(list)
    for s in scans:
        conf_by_type[s[6]].append(s[7])
    avg_conf = {k: round(sum(v)/len(v), 2) for k, v in conf_by_type.items()}

    return flask.render_template('stats.html',
        total=total,
        result_counts=json.dumps(result_counts),
        trend_dates=json.dumps(trend_dates),
        trend_conf=json.dumps(trend_conf),
        trend_labels=json.dumps(trend_labels),
        tumor_count=tumor_count,
        normal_count=normal_count,
        avg_conf=json.dumps(avg_conf),
    )

# ── Routes: Patient Auth ──────────────────────────────────────────────────────
@app.route('/patient/register', methods=['GET', 'POST'])
def patient_register():
    if flask.request.method == 'GET':
        return flask.render_template('patient_register.html')
    name     = flask.request.form.get('name', '')
    age      = flask.request.form.get('age', '')
    email    = flask.request.form.get('email', '')
    password = flask.request.form.get('password', '')
    if register_patient(name, age, email, password):
        flask.session['patient_email'] = email
        flask.session['patient_name']  = name
        return flask.redirect(flask.url_for('patient_dashboard'))
    return flask.render_template('patient_register.html', error='Email already registered.')

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if flask.request.method == 'GET':
        return flask.render_template('patient_login.html')
    email    = flask.request.form.get('email', '')
    password = flask.request.form.get('password', '')
    patient  = get_patient_by_email(email)
    if patient and patient[4] == password:
        flask.session['patient_email'] = email
        flask.session['patient_name']  = patient[1]
        return flask.redirect(flask.url_for('patient_dashboard'))
    return flask.render_template('patient_login.html', error='Invalid credentials.')

@app.route('/patient/logout')
def patient_logout():
    flask.session.pop('patient_email', None)
    flask.session.pop('patient_name', None)
    return flask.redirect(flask.url_for('main'))

@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    email  = flask.session['patient_email']
    scans  = get_patient_scans(email)
    recent = scans[:10][::-1]
    trend_dates = [s[4] for s in recent]
    trend_conf  = [round(s[7], 2) for s in recent]
    return flask.render_template('patient_dashboard.html',
        scans=scans,
        patient_name=flask.session.get('patient_name'),
        trend_dates=json.dumps(trend_dates),
        trend_conf=json.dumps(trend_conf),
    )

# ── Routes: Doctor Auth & Dashboard ──────────────────────────────────────────
@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if flask.request.method == 'GET':
        return flask.render_template('doctor_login.html')
    username = flask.request.form.get('username', '')
    password = flask.request.form.get('password', '')
    if username == DOCTOR_USERNAME and password == DOCTOR_PASSWORD:
        flask.session['is_doctor'] = True
        return flask.redirect(flask.url_for('doctor_dashboard'))
    return flask.render_template('doctor_login.html', error='Invalid credentials.')

@app.route('/doctor/logout')
def doctor_logout():
    flask.session.pop('is_doctor', None)
    return flask.redirect(flask.url_for('main'))

@app.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    from collections import Counter
    scans        = get_all_scans()
    total        = len(scans)
    tumor_count  = sum(1 for s in scans if s[6] not in ('No Tumor', 'Benign'))
    result_dist  = dict(Counter(s[6] for s in scans))
    return flask.render_template('doctor_dashboard.html',
        scans=scans[:15], total=total,
        tumor_count=tumor_count,
        result_dist=json.dumps(result_dist),
    )

@app.route('/doctor/note', methods=['POST'])
@doctor_required
def add_note():
    scan_id = flask.request.form.get('scan_id')
    note    = flask.request.form.get('note', '').strip()
    if scan_id and note:
        save_note(int(scan_id), note)
    return flask.redirect(flask.url_for('scan_detail', scan_id=scan_id))

# ── Routes: PDF Download ──────────────────────────────────────────────────────
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    scan_id      = flask.request.form.get('scan_id')
    patient_name = flask.request.form.get('patient_name', 'Unknown')
    patient_age  = flask.request.form.get('patient_age', 'N/A')
    scan_type    = flask.request.form.get('scan_type', 'brain')
    result       = flask.request.form.get('result', '')
    confidence   = float(flask.request.form.get('confidence', 0))
    scan_date    = flask.request.form.get('scan_date', '')
    image_path   = flask.request.form.get('image_path', '')
    all_conf     = [float(x) for x in flask.request.form.getlist('all_confidences')]
    labels       = BRAIN_LABELS if scan_type == 'brain' else SKIN_LABELS
    notes        = get_notes(int(scan_id)) if scan_id else []

    pdf = generate_pdf(patient_name, patient_age, scan_type, result, confidence,
                       all_conf, labels, image_path, scan_date, notes)
    return flask.send_file(pdf, as_attachment=True,
                           download_name=f'report_{patient_name}_{scan_date[:10]}.pdf',
                           mimetype='application/pdf')

# ── Routes: Debug ─────────────────────────────────────────────────────────────
@app.route('/debug', methods=['POST'])
def debug():
    file = flask.request.files.get('file')
    if not file:
        return flask.jsonify({"error": "No file provided"}), 400
    scan_type = flask.request.form.get('scan_type', 'brain')
    img_bytes = file.read()
    model  = brain_model if scan_type == 'brain' else skin_model
    labels = BRAIN_LABELS if scan_type == 'brain' else SKIN_LABELS
    tensor = preprocess_image(img_bytes).to(device)
    with torch.no_grad():
        y_hat = model(tensor)
        probs = softmax(y_hat, dim=1)[0]
    return flask.jsonify({
        "raw_logits":      [round(float(x), 4) for x in y_hat[0]],
        "probabilities":   {labels[i]: round(float(probs[i])*100, 2) for i in range(len(labels))},
        "predicted_class": labels[int(argmax(probs))],
        "note": "If all probabilities are ~25%, model weights may not have loaded correctly."
    })

@app.errorhandler(500)
def server_error(error):
    return flask.render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)