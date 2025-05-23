from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
import random
import pandas as pd
import re
from werkzeug.utils import secure_filename
import logging
import chardet

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'json', 'jsonl'}
app.secret_key = 'your-secret-key'  # Cần cho session
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Hàm kiểm tra định dạng file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Hàm kiểm tra encoding của file
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    confidence = result['confidence']
    logger.info(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence})")
    return encoding

# Tính năng 1: Convert Label for Model
def tokenize_with_positions(text):
    tokens = []
    for match in re.finditer(r'\S+', text):
        token = match.group()
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))
    return tokens

def label_tokens(text, labels):
    tokens = tokenize_with_positions(text)
    token_labels = []
    for token, start, end in tokens:
        assigned_label = 'O'
        for label_start, label_end, label_name in labels:
            if start >= label_start and end <= label_end:
                assigned_label = label_name
                break
        token_labels.append((token, assigned_label))
    return token_labels

def preprocess_text_and_labels(text, labels):
    new_text = ''
    mapping = {}
    new_idx = 0
    i = 0
    while i < len(text):
        char = text[i]
        prev_char = text[i - 1] if i > 0 else ''
        next_char = text[i + 1] if i + 1 < len(text) else ''
        if re.match(r'\w', char):
            new_text += char
            mapping[i] = new_idx
            new_idx += 1
        elif char.isspace():
            new_text += char
            mapping[i] = new_idx
            new_idx += 1
        else:
            if re.match(r'\w', prev_char) and re.match(r'\w', next_char):
                if new_text and new_text[-1] != ' ':
                    new_text += ' '
                    new_idx += 1
                new_text += char
                mapping[i] = new_idx
                new_idx += 1
                if next_char != ' ':
                    new_text += ' '
                    new_idx += 1
            elif not prev_char.isspace() and not next_char.isspace():
                if new_text and new_text[-1] != ' ':
                    new_text += ' '
                    new_idx += 1
                new_text += char
                mapping[i] = new_idx
                new_idx += 1
                if i + 1 < len(text) and text[i + 1] != ' ':
                    new_text += ' '
                    new_idx += 1
            elif re.match(r'\w', next_char):
                new_text += char
                mapping[i] = new_idx
                new_idx += 1
                if next_char != ' ':
                    new_text += ' '
                    new_idx += 1
            elif re.match(r'\w', prev_char):
                if new_text and new_text[-1] != ' ':
                    new_text += ' '
                    new_idx += 1
                new_text += char
                mapping[i] = new_idx
                new_idx += 1
            else:
                new_text += char
                mapping[i] = new_idx
                new_idx += 1
        i += 1
    new_labels = []
    for start, end, label in labels:
        try:
            new_start = mapping[start]
            new_end = mapping[end - 1] + 1
            new_labels.append([new_start, new_end, label])
        except KeyError:
            pass
    return new_text.strip(), new_labels

def convert_to_bio_format(token_label_pairs):
    tokens = []
    bio_labels = []
    prev_label = None
    for token, label in token_label_pairs:
        tokens.append(token)
        if label == "O":
            bio_labels.append("O")
            prev_label = None
            continue
        sentiment = label.upper()
        if prev_label == label:
            prefix = "I"
        else:
            prefix = "B"
        bio_labels.append(f"{prefix}-{sentiment}")
        prev_label = label
    return {"tokens": tokens, "bio_labels": bio_labels}

def convert_label_for_model(input_file):
    if not os.path.exists(input_file):
        logger.error(f"File không tồn tại: {input_file}")
        raise FileNotFoundError(f"File không tồn tại: {input_file}")
    encoding = detect_file_encoding(input_file)
    output_dir = "./preprocessed_data/converted_files"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_file)
    output_filename = filename.replace('.jsonl', '_bio.jsonl')
    output_file = os.path.join(output_dir, output_filename)
    try:
        df = pd.read_json(input_file, lines=True, encoding='utf-8')
    except UnicodeDecodeError:
        logger.error(f"Lỗi encoding khi đọc {input_file}, thử với encoding {encoding}")
        df = pd.read_json(input_file, lines=True, encoding=encoding)
    df[['pre_text', 'pre_labels']] = df.apply(
        lambda row: pd.Series(preprocess_text_and_labels(row['text'], row['labels'])), axis=1)
    df['token_label_pairs'] = df.apply(lambda row: label_tokens(row['pre_text'], row['pre_labels']), axis=1)
    json_list = [convert_to_bio_format(pairs) for pairs in df['token_label_pairs']]
    with open(output_file, "w", encoding="utf-8") as f:
        for item in json_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_file

# Tính năng 2: Split Train Annotator
def split_train_annotator(input_file):
    if not os.path.exists(input_file):
        logger.error(f"File không tồn tại: {input_file}")
        raise FileNotFoundError(f"File không tồn tại: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    selected_sentences = data[:600]
    output_dir = "./preprocessed_data/split_predictions"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(12):
        subset = selected_sentences[i * 50:(i + 1) * 50]
        filename = os.path.join(output_dir, f"predictions_part_{i+1}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=4)
    return output_dir

# Tính năng 3: Split Final Data
def split_final_data(input_file):
    if not os.path.exists(input_file):
        logger.error(f"File không tồn tại: {input_file}")
        raise FileNotFoundError(f"File không tồn tại: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:1000]
    names = ["Minh", "Vu", "Binh", "Long"]
    base_dir = "./preprocessed_data/final_data"
    os.makedirs(base_dir, exist_ok=True)
    for part_idx, name in enumerate(names):
        part_data = data[part_idx * 250: (part_idx + 1) * 250]
        person_dir = os.path.join(base_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        for sub_idx in range(5):
            sub_data = part_data[sub_idx * 50: (sub_idx + 1) * 50]
            file_path = os.path.join(person_dir, f"{name}_set{sub_idx+1}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sub_data, f, ensure_ascii=False, indent=2)
    return base_dir

# Hàm lấy danh sách file và thư mục
def get_file_structure():
    base_dir = os.path.abspath("./preprocessed_data")  # Đường dẫn tuyệt đối
    structure = {"final_data": {}, "split_predictions": [], "converted_files": []}
    final_data_dir = os.path.join(base_dir, "final_data")
    split_predictions_dir = os.path.join(base_dir, "split_predictions")
    converted_files_dir = os.path.join(base_dir, "converted_files")

    if os.path.exists(final_data_dir):
        for person in os.listdir(final_data_dir):
            person_path = os.path.join(final_data_dir, person)
            if os.path.isdir(person_path):
                structure["final_data"][person] = [
                    f for f in os.listdir(person_path) if f.endswith(".json")
                ]
    if os.path.exists(split_predictions_dir):
        structure["split_predictions"] = [
            f for f in os.listdir(split_predictions_dir) if f.endswith(".json")
        ]
    if os.path.exists(converted_files_dir):
        structure["converted_files"] = [
            f for f in os.listdir(converted_files_dir) if f.endswith(".jsonl")
        ]
    return structure

# Hàm đọc nội dung file JSON/JSONL
def read_json_file(file_path):
    try:
        if not os.path.exists(file_path):
            logger.error(f"File không tồn tại: {file_path}")
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
        encoding = detect_file_encoding(file_path)
        if file_path.endswith('.jsonl'):
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except UnicodeDecodeError:
        logger.error(f"Lỗi encoding khi đọc {file_path}, thử với encoding {encoding}")
        with open(file_path, "r", encoding=encoding) as f:
            return [json.loads(line) for line in f] if file_path.endswith('.jsonl') else json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        return {"error": str(e)}

# Route trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    structure = get_file_structure()
    message = None
    if request.method == 'POST':
        if 'file' not in request.files:
            message = "Không có file được tải lên"
            logger.warning("Không có file được tải lên")
        else:
            file = request.files['file']
            feature = request.form.get('feature')
            if file.filename == '':
                message = "Chưa chọn file"
                logger.warning("Chưa chọn file")
            elif allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(file_path)
                    logger.info(f"File đã được lưu: {file_path}")
                    if feature == 'convert':
                        if not file_path.endswith('.jsonl'):
                            message = "Tính năng Convert Label for Model chỉ chấp nhận file .jsonl"
                            logger.warning(f"File {file_path} không phải .jsonl")
                        else:
                            # Lưu file_path vào session và hiển thị dữ liệu
                            session['input_file_path'] = file_path
                            content = read_json_file(file_path)
                            if isinstance(content, dict) and 'error' in content:
                                message = f"Lỗi đọc file: {content['error']}"
                            else:
                                return render_template('preview.html', content=content, file_path=file_path)
                    elif feature == 'split_annotator':
                        if not file_path.endswith('.json'):
                            message = "Tính năng Split Train Annotator chỉ chấp nhận file .json"
                            logger.warning(f"File {file_path} không phải .json")
                        else:
                            output = split_train_annotator(file_path)
                            message = f"Đã chia file vào thư mục: {output}"
                            logger.info(f"Đã xử lý split_annotator: {output}")
                    elif feature == 'split_final':
                        if not file_path.endswith('.json'):
                            message = "Tính năng Split Final Data chỉ chấp nhận file .json"
                            logger.warning(f"File {file_path} không phải .json")
                        else:
                            output = split_final_data(file_path)
                            message = f"Đã chia file vào thư mục: {output}"
                            logger.info(f"Đã xử lý split_final: {output}")
                    structure = get_file_structure()
                except Exception as e:
                    message = f"Lỗi: {str(e)}"
                    logger.error(f"Lỗi khi xử lý file {file_path}: {str(e)}")
            else:
                message = "Định dạng file không hỗ trợ, chỉ chấp nhận .json hoặc .jsonl"
                logger.warning(f"Định dạng file không hỗ trợ: {filename}")
    return render_template('index.html', structure=structure, message=message)

# Route xem trước dữ liệu đầu vào
@app.route('/preview', methods=['POST'])
def preview():
    action = request.form.get('action')
    file_path = session.get('input_file_path')
    if not file_path or not os.path.exists(file_path):
        return render_template('error.html', message="File đầu vào không tồn tại")
    if action == 'process':
        try:
            output = convert_label_for_model(file_path)
            session.pop('input_file_path', None)  # Xóa session
            structure = get_file_structure()
            return render_template('index.html', structure=structure, message=f"Đã xử lý: {output}")
        except Exception as e:
            return render_template('error.html', message=f"Lỗi khi xử lý: {str(e)}")
    else:  # Hủy
        session.pop('input_file_path', None)
        return redirect(url_for('index'))

# Route xem nội dung file
@app.route('/view/<path:file_path>')
def view_file(file_path):
    content = read_json_file(file_path)
    if isinstance(content, dict) and 'error' in content:
        return render_template('error.html', message=content['error'])
    return render_template('view.html', content=content, file_path=file_path, is_jsonl=file_path.endswith('.jsonl'))

if __name__ == '__main__':
    app.run(debug=True)