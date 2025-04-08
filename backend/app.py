from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# To test the API
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask!"})

# Base upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the base upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_images():
    # Get restaurant name from form data
    restaurant_name = request.form.get("restaurantName")
    
    if not restaurant_name:
        return jsonify({"message": "Restaurant name is required"}), 400

    # Create a directory for the restaurant (use secure_filename to sanitize input)
    restaurant_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(restaurant_name))
    if not os.path.exists(restaurant_folder):
        os.makedirs(restaurant_folder)

    # Get uploaded files
    files = request.files.getlist("images")
    
    if not files:
        return jsonify({"message": "No images uploaded"}), 400

    saved_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(restaurant_folder, filename)
            file.save(filepath)
            saved_files.append(filename)

    return jsonify({
        "message": f"{len(saved_files)} images uploaded successfully for '{restaurant_name}'!",
        "files": saved_files,
        "restaurant": restaurant_name,
        "folder_path": restaurant_folder,
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)