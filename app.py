from flask import Flask, render_template, Response
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

user_logged_in = True  # Toggle login state

def generate_frames():
    camera = None
    try:
        camera = cv2.VideoCapture(0)  # Webcam access
        if not camera.isOpened():
            raise Exception("Could not access the webcam")
            
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:            
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {str(e)}")
        yield b''
    finally:
        if camera is not None:
            camera.release()

@app.route('/')
def home():
    return render_template('home.html', user_logged_in=user_logged_in)

@app.route('/object')
def object():
    return render_template('object.html', user_logged_in=user_logged_in)

@app.route('/object/video_feed')
def object_video_feed():
    def generate_object_frames():
        cap = None
        try:
            # Use os.path.join for cross-platform compatibility
            import os
            weights_path = os.path.join("python_Scripts", "yolov3.weights")
            config_path = os.path.join("python_Scripts", "yolov3.cfg")
            names_path = os.path.join("python_Scripts", "coco.names")
            
            # Check if model files exist
            if not all(os.path.exists(p) for p in [weights_path, config_path, names_path]):
                raise Exception("Required model files are missing")

            net = cv2.dnn.readNet(weights_path, config_path)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

            with open(names_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not access the webcam")

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break
                
                # Prepare the frame for YOLO
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                # Initialize lists for detection results
                boxes = []
                confidences = []
                class_ids = []

                height, width, channels = frame.shape

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        # Set a confidence threshold (0.5)
                        if confidence > 0.5:  
                            # Get the bounding box
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # Apply non-maxima suppression to eliminate redundant overlapping boxes
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                # Draw rectangles around detected objects and label them
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        label = classes[class_ids[i]]  # Get the class name from the label
                        confidence = confidences[i]

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
                        cv2.putText(frame, f"{label} ({round(confidence * 100, 2)}%)", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Encode frame to bytes
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            # When everything is done, release the capture and close windows
            cap.release()
        except Exception as e:
            print(f"Error in object_video_feed: {str(e)}")
            yield b''
        finally:
            if cap is not None:
                cap.release()

    return Response(generate_object_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/emotion')
def emotion():
    return render_template('Emotion.html', user_logged_in=user_logged_in)


@app.route('/emotion/video_feed')
def emotion_video_feed():
    def generate_emotion_frame():
        cap = None
        try:
            # Load both face and eye cascade classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not access webcam")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Get regions of interest for eyes and smile
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    
                    # Detect eyes
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    
                    # Detect smile
                    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
                    
                    # Determine emotion based on eyes and smile
                    if len(eyes) >= 2:  # Both eyes detected
                        if len(smiles) > 0:  # Smile detected
                            emotion = "Happy"
                            color = (0, 255, 0)  # Green
                        else:
                            emotion = "Neutral"
                            color = (255, 255, 0)  # Yellow
                    else:
                        emotion = "Unknown"
                        color = (0, 0, 255)  # Red
                    
                    # Draw rectangles for eyes and smile
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                    
                    # Display emotion
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Encode frame to bytes
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    break
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error in emotion_video_feed: {str(e)}")
            yield b''
        finally:
            if cap is not None:
                cap.release()

    return Response(generate_emotion_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/human')
def human():
    return render_template('human.html', user_logged_in=user_logged_in)

@app.route('/human/video_feed')
def human_video_feed():
    def generate_human_frames():
        # Load pre-trained YOLO model (weights and config)
        net = cv2.dnn.readNet("python_Scripts/yolov3.weights", "python_Scripts/yolov3.cfg")
        with open("python_Scripts/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prepare frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Analyze detections
            human_count = 0
            height, width, _ = frame.shape
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and class_id == 0:  # 0 corresponds to 'person' class
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Draw rectangles around detected humans
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    human_count += 1

            # Display the human count
            cv2.putText(frame, f'Humans detected: {human_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_human_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign')
def sign():
    return render_template('sign.html', user_logged_in=user_logged_in)

@app.route('/sign/video_feed')
def sign_video_feed():
    def generate_sign_frames():
        pass
    return Response(generate_sign_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vehicle')
def vehicle():
    return render_template('vehicle.html', user_logged_in=user_logged_in)

@app.route('/vehicle/video_feed')
def vehicle_video_feed():
    def generate_vehicle_frames():
        net = cv2.dnn.readNet("python_Scripts/yolov3.weights", "python_Scripts/yolov3.cfg")  
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        with open("python_Scripts/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Open the webcam (0 is the default camera)
        cap = cv2.VideoCapture(0)

        # Check if the webcam was opened correctly
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Analyze the detections
            vehicle_count = 0
            height, width, channels = frame.shape
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and (class_id in (2,3,5,7)):  # 0 corresponds to 'car','motorbike','bus','truck' classes in COCO dataset
                        # Get the bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maxima suppression to eliminate redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw rectangles around detected humans
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    vehicle_count += 1

            # Display the human count
            cv2.putText(frame, f'Vehicles detected: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        # When everything is done, release the capture and close windows
        cap.release()

    return Response(generate_vehicle_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/movement')
def movement():
    return render_template('movement.html', user_logged_in=user_logged_in)

@app.route('/movement/video_feed')
def movement_video_feed():
    def generate_movement_frames():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return

        # Initialize the MOG2 background subtractor
        mog_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from the webcam.")
                    break

                # Resize the frame for faster processing
                frame_resized = cv2.resize(frame, (640, 480))

                # Apply Gaussian Blur to reduce noise
                blurred_frame = cv2.GaussianBlur(frame_resized, (5, 5), 0)

                # Apply the MOG2 background subtraction
                learning_rate = 0.01
                foreground_mask = mog_subtractor.apply(blurred_frame, learningRate=learning_rate)

                # Apply binary thresholding to refine the mask
                _, thresholded_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)

                # Morphological operations to further clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                clean_mask = cv2.morphologyEx(thresholded_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours for object detection
                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:  # Filter by area
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Encode frame to bytes
                ret, buffer = cv2.imencode('.jpg', clean_mask)
                if not ret:
                    print("Error: Could not encode frame.")
                    break

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            # Ensure the webcam is released even if an exception occurs
            cap.release()

    return Response(generate_movement_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
