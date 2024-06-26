from flask import Flask, request, render_template, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/processed_video'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def bg_movement(frame, threshold, kernel, object_detector, roi_x, roi_y, roi_height, roi_width):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    mask = object_detector.apply(blurred_frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    roi_mask = mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    movement = cv2.countNonZero(roi_mask)
    return movement > threshold

def movement_detection(input_video_path, output_video_path, model):
    print("loading model")
    
    cap = cv2.VideoCapture(input_video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x = 250
    roi_y = 200
    roi_width = 500
    roi_height = frame_height - roi_y

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    object_detector = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    video_writer = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))

    initial_height = None
    previous_height = None

    for frame_index in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        movement = bg_movement(frame, 9000, kernel, object_detector, roi_x, roi_y, roi_height, roi_width)

        text_x = frame_width - 200
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)
        max_area = 0
        max_box = None

        for result in results.xyxy[0]:
            x_min, y_min, x_max, y_max, confidence, class_id = result.tolist()
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                max_box = (x_min, y_min, x_max, y_max)

        center_x = center_y = height = height_diff = 0

        if max_box:
            x_min, y_min, x_max, y_max = [int(coord) for coord in max_box]
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            height = y_max - y_min

            if initial_height is None:
                initial_height = height

            height_diff = abs(height - initial_height)
            state = 0
            if previous_height is not None:
                continuous_height_diff = abs(height - previous_height)
                if continuous_height_diff > 5 and movement:
                    state = 1
                if height_diff < 5 and not movement:
                    state = 0
            else:
                if height_diff < 5 and not movement:
                    state = 0
                if height_diff > 5 or movement:
                    state = 1
                
            previous_height = height

        print(f"Frame {frame_index} : {state}")

        cv2.putText(frame, f'State: {state}', (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        if max_box:
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Height: {int(height)}', (center_x, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
        
        video_writer.write(frame)

    cap.release()
    video_writer.release()

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        output_filename = 'processed_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        movement_detection(filepath, output_filepath, model)

        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        if ret:
            thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnail_' + filename + '.jpg')
            cv2.imwrite(thumbnail_path, frame)
        cap.release()

        return jsonify({'filename': output_filename, 'thumbnail': 'thumbnail_' + filename + '.jpg'}), 200
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
