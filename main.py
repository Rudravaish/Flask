from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from model import predict_lesion
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-change-in-production")

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cleanup_old_uploads():
    """Clean up uploaded files older than 1 hour to prevent storage issues."""
    try:
        import time
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                # Remove files older than 1 hour (3600 seconds)
                if file_age > 3600:
                    os.remove(filepath)
                    app.logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        app.logger.warning(f"Error during cleanup: {str(e)}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define Fitzpatrick skin types with detailed descriptions
FITZPATRICK_TYPES = {
    "I": "Very fair - Always burns, never tans (Celtic, Scandinavian)",
    "II": "Fair - Usually burns, tans poorly (Northern European)", 
    "III": "Medium - Sometimes mild burn, tans uniformly (Mixed European)",
    "IV": "Olive - Burns minimally, always tans well (Mediterranean, Asian)",
    "V": "Brown - Rarely burns, tans profusely (Middle Eastern, Latino)",
    "VI": "Dark brown/Black - Never burns, deeply pigmented (African, Aboriginal)"
}

# Body types with melanoma risk factors
BODY_TYPES = {
    "average": "Average risk profile",
    "fair_many_moles": "Fair skin with many moles (50+ moles)",
    "family_history": "Family history of melanoma",
    "previous_skin_cancer": "Previous skin cancer diagnosis",
    "immunocompromised": "Immunocompromised or taking immunosuppressive drugs",
    "frequent_sun_exposure": "Frequent sun exposure or history of severe sunburns"
}

# Skin types with potential model bias
BIAS_WARNING_TYPES = ["V", "VI"]


@app.route('/', methods=['GET', 'POST'])
def home():
    """Main route for file upload and prediction."""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'image' not in request.files:
                flash('No file selected. Please choose an image to upload.', 'error')
                return redirect(request.url)
            
            file = request.files['image']
            
            # Get form parameters
            skin_type = request.form.get('skin_type', 'III')  # Default to Type III
            body_type = request.form.get('body_type', 'average')  # Default to average risk
            
            # Evolution tracking (E in ABCDE)
            has_evolved = request.form.get('has_evolved') == 'yes'
            evolution_weeks = int(request.form.get('evolution_weeks', 0)) if request.form.get('evolution_weeks') else 0
            
            # Manual measurements (optional)
            manual_length = float(request.form.get('manual_length', 0)) if request.form.get('manual_length') else None
            manual_width = float(request.form.get('manual_width', 0)) if request.form.get('manual_width') else None
            
            # Check if file was actually selected
            if not file.filename or file.filename == '':
                flash('No file selected. Please choose an image to upload.', 'error')
                return redirect(request.url)
            
            # Validate file type
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WebP).', 'error')
                return redirect(request.url)
            
            if file and file.filename and allowed_file(file.filename):
                # Secure the filename - we know filename is not None at this point
                filename = secure_filename(str(file.filename))
                
                # Create unique filename to avoid conflicts
                import time
                timestamp = str(int(time.time()))
                name, ext = os.path.splitext(filename)
                unique_filename = f"{name}_{timestamp}{ext}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save the file
                file.save(filepath)
                app.logger.info(f"File saved to: {filepath}")
                
                # Make prediction with enhanced parameters
                try:
                    prediction_result = predict_lesion(
                        filepath, 
                        skin_type=skin_type,
                        body_type=body_type,
                        has_evolved=has_evolved,
                        evolution_weeks=evolution_weeks,
                        manual_length=manual_length,
                        manual_width=manual_width
                    )
                    
                    # Handle different return formats safely
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 3:
                        prediction, confidence, analysis_data = prediction_result
                    elif isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                        prediction, confidence = prediction_result
                        analysis_data = None
                    else:
                        # Fallback for unexpected formats
                        prediction = "Error in Analysis"
                        confidence = 0
                        analysis_data = None
                    
                    app.logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
                    
                    # Generate detailed analysis summary
                    analysis_summary = None
                    if analysis_data:
                        analysis_summary = {
                            'detected_skin_tone': analysis_data.get('detected_skin_tone', skin_type),
                            'features': {
                                'asymmetry': analysis_data.get('asymmetry', 0),
                                'border': analysis_data.get('border', 0),
                                'color': analysis_data.get('color', 0),
                                'diameter': analysis_data.get('diameter', 0)
                            },
                            'analysis_type': analysis_data.get('analysis_type', 'standard')
                        }
                    
                    # Check for bias warning
                    bias_warning = None
                    if skin_type in BIAS_WARNING_TYPES:
                        bias_warning = f"Model may have reduced accuracy on Fitzpatrick skin type {skin_type} ({FITZPATRICK_TYPES[skin_type]}). Please consult a dermatologist for professional evaluation."
                    
                    # Enhanced bias warning for darker skin tones with better analysis
                    detected_skin_tone = analysis_data.get('detected_skin_tone') if analysis_data else skin_type
                    if detected_skin_tone in ['V', 'VI'] and analysis_data:
                        enhanced_warning = f"Analysis optimized for darker skin tone (Type {detected_skin_tone}). This analysis accounts for melanin-specific patterns and post-inflammatory changes common in darker skin."
                    else:
                        enhanced_warning = None
                    
                    # Clean up old uploaded files to prevent storage issues
                    cleanup_old_uploads()
                    
                    return render_template('index.html', 
                                         result=prediction, 
                                         confidence=confidence, 
                                         image_path=filepath,
                                         filename=unique_filename,
                                         skin_type=skin_type,
                                         skin_type_description=FITZPATRICK_TYPES[skin_type],
                                         bias_warning=bias_warning,
                                         enhanced_warning=enhanced_warning,
                                         analysis_summary=analysis_summary,
                                         detected_skin_tone=detected_skin_tone,
                                         fitzpatrick_types=FITZPATRICK_TYPES)
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    flash('Error processing image. Please try again with a different image.', 'error')
                    # Clean up the uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return redirect(request.url)
                    
        except Exception as e:
            app.logger.error(f"Upload error: {str(e)}")
            flash('An error occurred while processing your upload. Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('index.html', fitzpatrick_types=FITZPATRICK_TYPES)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    app.logger.error(f"Internal server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any uncaught exceptions."""
    app.logger.error(f"Unhandled exception: {str(e)}")
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
