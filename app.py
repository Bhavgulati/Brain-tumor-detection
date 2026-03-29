import flask
import sqlite3
import os
import io
import datetime
from io import BytesIO
from torch import argmax, load, softmax
from torch import device as DEVICE
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import resnet50
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER

UPLOAD_FOLDER = os.path.join('static', 'photos')
DB_PATH = 'patient_history.db'

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "brain_tumor_secret_2024"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ── FIX 1: Corrected label order to match standard Kaggle Brain Tumor dataset ──
# Original order was: ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary']
# Correct order is:   ['glioma', 'meningioma', 'notumor', 'pituitary']
LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

LABEL_INFO = {
    'No Tumor':   'No tumor was detected in the MRI scan. The brain tissue appears normal.',
    'Meningioma': 'A typically slow-growing tumor that forms on the membranes covering the brain and spinal cord. Usually benign and treatable.',
    'Glioma':     'A tumor that originates in the glial cells of the brain or spinal cord. Requires immediate medical attention.',
    'Pituitary':  'A tumor that forms in the pituitary gland at the base of the brain. Usually benign and often manageable.',
}
LABEL_SEVERITY = {
    'No Tumor': 'normal', 'Meningioma': 'warning', 'Glioma': 'danger', 'Pituitary': 'warning'
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS scans (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT NOT NULL,
        patient_age  TEXT,
        scan_date    TEXT NOT NULL,
        result       TEXT NOT NULL,
        confidence   REAL NOT NULL,
        image_path   TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

def save_scan(patient_name, patient_age, result, confidence, image_path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO scans (patient_name,patient_age,scan_date,result,confidence,image_path) VALUES (?,?,?,?,?,?)',
        (patient_name, patient_age, datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), result, confidence, image_path)
    )
    conn.commit()
    conn.close()

def get_all_scans():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('SELECT * FROM scans ORDER BY id DESC').fetchall()
    conn.close()
    return rows

# ── Model ─────────────────────────────────────────────────────────────────────
device = "cuda" if is_available() else "cpu"
resnet_model = resnet50(weights=None)
n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(
    Linear(n_inputs, 2048), SELU(), Dropout(p=0.4),
    Linear(2048, 2048),     SELU(), Dropout(p=0.4),
    Linear(2048, 4)
)
resnet_model.to(device)
resnet_model.load_state_dict(load('./models/bt_resnet50_model.pt', map_location=DEVICE(device)))
resnet_model.eval()

# ── FIX 2: Corrected image size from 512x512 to 224x224 (standard ResNet input)
# ── FIX 3: Added ImageNet normalization that ResNet50 expects
def preprocess_image(image_bytes):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                  std=[0.229, 0.224, 0.225])      # ImageNet std
    ])
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes)
    y_hat  = resnet_model(tensor.to(device))
    probs  = softmax(y_hat, dim=1)[0]
    class_id = int(argmax(probs))
    confidence_list = [round(float(p) * 100, 2) for p in probs]
    return class_id, LABELS[class_id], confidence_list

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ── PDF ───────────────────────────────────────────────────────────────────────
def generate_pdf(patient_name, patient_age, result, confidence, all_confidences, image_path, scan_date):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch,   bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_s   = ParagraphStyle('T', parent=styles['Title'],   fontSize=20, textColor=colors.HexColor('#1a365d'), spaceAfter=4,  alignment=TA_CENTER)
    sub_s     = ParagraphStyle('S', parent=styles['Normal'],  fontSize=11, textColor=colors.HexColor('#4a5568'), spaceAfter=16, alignment=TA_CENTER)
    section_s = ParagraphStyle('H', parent=styles['Normal'],  fontSize=13, fontName='Helvetica-Bold', textColor=colors.HexColor('#2d3748'), spaceAfter=8)
    disc_s    = ParagraphStyle('D', parent=styles['Normal'],  fontSize=9,  textColor=colors.HexColor('#718096'),
                                borderColor=colors.HexColor('#e2e8f0'), borderWidth=1, borderPadding=8, backColor=colors.HexColor('#f7fafc'))

    story.append(Paragraph("Brain Tumor Detection Report", title_s))
    story.append(Paragraph("AI-Powered Medical Imaging Analysis — For Research Use Only", sub_s))
    story.append(Table([['']], colWidths=[6.5*inch],
                        style=TableStyle([('LINEABOVE',(0,0),(0,0),1.5,colors.HexColor('#3182ce'))])))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Patient Information", section_s))
    info_table = Table([
        ['Patient Name', patient_name or 'N/A'],
        ['Age',          patient_age  or 'N/A'],
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
    dc = '#276749' if result == 'No Tumor' else '#9b2c2c'
    diag_table = Table([
        ['Detected Condition', result],
        ['Confidence',         f'{confidence:.1f}%'],
        ['Description',        LABEL_INFO.get(result, '')],
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
    prob_data = [['Tumor Type', 'Confidence (%)']]
    for label, conf in zip(LABELS, all_confidences):
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

    if image_path and os.path.exists(image_path):
        story.append(Paragraph("MRI Scan", section_s))
        try:
            story.append(RLImage(image_path, width=3*inch, height=3*inch))
        except Exception:
            pass
        story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an AI system for research and educational "
        "purposes only. It is not a substitute for professional medical advice. Always consult a qualified healthcare provider.", disc_s))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def main():
    scans = get_all_scans()
    total = len(scans)
    tumor_count = sum(1 for s in scans if s[4] != 'No Tumor')
    return flask.render_template('DiseaseDet.html', total=total, tumor_count=tumor_count, recent=scans[:5])

@app.route('/uimg', methods=['GET', 'POST'])
def uimg():
    if flask.request.method == 'GET':
        return flask.render_template('uimg.html')

    file         = flask.request.files.get('file')
    patient_name = flask.request.form.get('patient_name', 'Unknown')
    patient_age  = flask.request.form.get('patient_age', 'N/A')

    if not file or not allowed_file(file.filename):
        return flask.redirect(flask.url_for('uimg'))

    img_bytes = file.read()
    class_id, class_name, all_confidences = get_prediction(img_bytes)
    top_conf  = all_confidences[class_id]
    scan_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    filename   = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    Image.open(BytesIO(img_bytes)).convert('RGB').save(image_path)

    save_scan(patient_name, patient_age, class_name, top_conf, image_path)

    return flask.render_template('pred.html',
        result=class_name,
        confidence=top_conf,
        all_confidences=all_confidences,
        labels=LABELS,
        description=LABEL_INFO.get(class_name, ''),
        severity=LABEL_SEVERITY.get(class_name, 'warning'),
        patient_name=patient_name,
        patient_age=patient_age,
        scan_date=scan_date,
        image_path=image_path,
        filename=filename,
    )

@app.route('/history')
def history():
    scans = get_all_scans()
    return flask.render_template('history.html', scans=scans)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    patient_name    = flask.request.form.get('patient_name', 'Unknown')
    patient_age     = flask.request.form.get('patient_age', 'N/A')
    result          = flask.request.form.get('result', '')
    confidence      = float(flask.request.form.get('confidence', 0))
    scan_date       = flask.request.form.get('scan_date', '')
    image_path      = flask.request.form.get('image_path', '')
    all_confidences = [float(x) for x in flask.request.form.getlist('all_confidences')]

    pdf = generate_pdf(patient_name, patient_age, result, confidence, all_confidences, image_path, scan_date)
    return flask.send_file(pdf, as_attachment=True,
                           download_name=f'report_{patient_name}_{scan_date[:10]}.pdf',
                           mimetype='application/pdf')

# ── FIX 4: Debug route to diagnose model output ───────────────────────────────
# Visit POST /debug with an image file to see raw probabilities
# Remove this route in production
@app.route('/debug', methods=['POST'])
def debug():
    file = flask.request.files.get('file')
    if not file:
        return flask.jsonify({"error": "No file provided"}), 400
    img_bytes = file.read()
    import torch
    tensor = preprocess_image(img_bytes)
    with torch.no_grad():
        y_hat = resnet_model(tensor.to(device))
        probs = softmax(y_hat, dim=1)[0]
    result = {
        "raw_logits": [round(float(x), 4) for x in y_hat[0]],
        "probabilities": {
            LABELS[i]: round(float(probs[i]) * 100, 2) for i in range(len(LABELS))
        },
        "predicted_class": LABELS[int(argmax(probs))],
        "note": "If all probabilities are near 25%, the model weights may not have loaded correctly."
    }
    return flask.jsonify(result)

@app.errorhandler(500)
def server_error(error):
    return flask.render_template('error.html'), 500

if __name__ == '__main__':
    app.run(debug=True)