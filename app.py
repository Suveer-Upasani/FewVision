import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
import shutil

# Import FewVision Pipeline modules
from main import process_image, process_folder, REPORTS_DIR
from report_generator import generate_image_report
from augmentations import generate_batch
import feature_extraction
from few_shot_model import PrototypicalNetwork

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Image Quality & Content Analysis
            result = process_image(filepath)
            
            # Generate the heatmap report
            report_path = generate_image_report(result, REPORTS_DIR)
            report_filename = os.path.basename(report_path)
            
            # 2. Few-Shot ML Pipeline (Option B)
            # a. Augment just this one image (generate 10 versions in temp dir)
            temp_aug_dir = os.path.join(app.config['UPLOAD_FOLDER'], "temp_aug")
            os.makedirs(temp_aug_dir, exist_ok=True)
            
            # Clear old temp augmentations
            for f in os.listdir(temp_aug_dir):
                os.remove(os.path.join(temp_aug_dir, f))
                
            generate_batch(filepath, output_dir=temp_aug_dir, num_images=10, augmentations=result.augmentations)
            
            # b. Extract Features
            features, labels = feature_extraction.extract_features(temp_aug_dir)
            
            # c. Few-Shot Prediction using SAVED prototypes
            
            model = PrototypicalNetwork()
            # Load existing prototypes
            if os.path.exists("prototypes.npy"):
                model.prototypes = np.load("prototypes.npy", allow_pickle=True).item()
                
                # Predict on all 10 augmentations and take majority vote
                predictions = model.predict_batch(features)
                pass_count = predictions.count("PASS")
                defect_count = predictions.count("DEFECT")
                
                if defect_count >= pass_count:
                    final_prediction = "DEFECT"
                    confidence = defect_count / len(predictions)
                else:
                    final_prediction = "PASS"
                    confidence = pass_count / len(predictions)
            else:
                final_prediction = "UNKNOWN"
                confidence = 0.0

            # Clean up temp ML files (optional, leaving them for now for debugging)
            
            return jsonify({
                'success': True,
                'metrics': {
                    'blur': round(result.quality.blur, 2),
                    'noise': round(result.quality.noise, 2),
                    'quality_score': round(result.quality.quality_score, 2),
                    'content_score': round(result.content.content_score, 2),
                    'suitability': round(result.suitability_score, 2),
                    'suitability_rating': result.suitability_rating
                },
                'ml_prediction': final_prediction,
                'ml_confidence': round(confidence * 100, 1),
                'report_url': f'/reports/{report_filename}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
        
    train_dir = os.path.join(app.config['UPLOAD_FOLDER'], "train_set")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    
    try:
        # Save all uploaded training files preserving class folders
        for file in files:
            if file:
                rel_path = file.filename.replace('\\', '/')
                parts = rel_path.split('/')
                
                if len(parts) > 1 and parts[-2] in {"PASS", "DEFECT"}:
                    class_label = parts[-2]
                    sub_filename = secure_filename(parts[-1])
                    dest_dir = os.path.join(train_dir, class_label)
                    os.makedirs(dest_dir, exist_ok=True)
                    file.save(os.path.join(dest_dir, sub_filename))
                else:
                    # Save flat if no subfolder layout detected
                    sub_filename = secure_filename(parts[-1])
                    file.save(os.path.join(train_dir, sub_filename))
        
        # Trigger the entire main pipeline (Quality -> Augment -> Extract -> Prototypical Network)
        # on the newly uploaded training directory. This will overwrite prototypes.npy with the new object!
        process_folder(train_dir)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
