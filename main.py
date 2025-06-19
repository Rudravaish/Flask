from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from basic_model import predict_lesion
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback-secret-key-for-dev")

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Fitzpatrick skin types
FITZPATRICK_TYPES = {
    'I': 'Type I',
    'II': 'Type II', 
    'III': 'Type III',
    'IV': 'Type IV',
    'V': 'Type V',
    'VI': 'Type VI'
}

def cleanup_old_uploads():
    """Clean up uploaded files older than 1 hour"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            current_time = time.time()
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
    except Exception as e:
        app.logger.warning(f"Cleanup failed: {e}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    """Main route for file upload and prediction"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Check file extension
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload a valid image file.', 'error')
                return redirect(request.url)
            
            # Get form data
            skin_type = request.form.get('skin_type', 'III')
            body_part = request.form.get('body_part', 'other')
            has_evolved = 'has_evolved' in request.form
            evolution_weeks = int(request.form.get('evolution_weeks', 0)) if has_evolved else 0
            
            # Save uploaded file
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = str(int(time.time()))
                unique_filename = f"{timestamp}_{filename}"
                
                # Create upload directory if it doesn't exist
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                app.logger.info(f"File saved to: {filepath}")
                
                # Make prediction
                try:
                    prediction, confidence, analysis_data = predict_lesion(
                        filepath, skin_type, body_part, has_evolved, evolution_weeks
                    )
                    
                    app.logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
                    
                    # Generate analysis summary
                    analysis_summary = None
                    if analysis_data:
                        analysis_summary = {
                            'detected_skin_tone': analysis_data.get('detected_skin_tone', f'Type {skin_type}'),
                            'features': {
                                'asymmetry': analysis_data.get('asymmetry', 0),
                                'border': analysis_data.get('border', 0),
                                'color': analysis_data.get('color', 0),
                                'diameter': analysis_data.get('diameter', 0)
                            },
                            'analysis_type': analysis_data.get('analysis_type', 'basic')
                        }
                    
                    # Clean up old files
                    cleanup_old_uploads()
                    
                    return render_template('index.html', 
                                         result=prediction, 
                                         confidence=confidence, 
                                         image_path=filepath,
                                         filename=unique_filename,
                                         skin_type=skin_type,
                                         skin_type_description=FITZPATRICK_TYPES[skin_type],
                                         analysis_summary=analysis_summary,
                                         detected_skin_tone=f'Type {skin_type}',
                                         fitzpatrick_types=FITZPATRICK_TYPES,
                                         body_part_options=['face', 'scalp', 'neck', 'chest', 'back', 'abdomen', 'arms', 'hands', 'legs', 'feet', 'other'])
                    
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    flash('Error processing image. Please try again with a different image.', 'error')
                    return redirect(request.url)
                    
        except Exception as e:
            app.logger.error(f"Upload error: {str(e)}")
            flash('An error occurred while processing your upload. Please try again.', 'error')
            return redirect(request.url)
    
    # GET request - show the form
    body_part_options = ['face', 'scalp', 'neck', 'chest', 'back', 'abdomen', 'arms', 'hands', 'legs', 'feet', 'other']
    return render_template('index.html', fitzpatrick_types=FITZPATRICK_TYPES, body_part_options=body_part_options)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    app.logger.error(f"Internal error: {str(e)}")
    return render_template('index.html', error="Internal server error"), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any uncaught exceptions"""
    app.logger.error(f"Unhandled exception: {str(e)}")
    return render_template('index.html', error="An unexpected error occurred"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)