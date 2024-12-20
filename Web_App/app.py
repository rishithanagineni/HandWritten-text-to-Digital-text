from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import keras_ocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np

app = Flask(__name__)

# Setup your model (same as in your original script)
class H2DPipeline:
    def __init__(self, model_path):
        self.detector = keras_ocr.detection.Detector()
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

    def __get_bounding_boxes_for_image(self, image_path):
        bboxes = self.detector.detect([image_path], link_threshold=0.4)
        return self.__sort_into_lines(bboxes[0])

    def __line_check(self, box1, box2):
        box1_y_start = min(box1[0, 1], box1[1, 1])
        box1_y_end = max(box1[2, 1], box1[3, 1])
        box2_y_start = min(box2[0, 1], box2[1, 1])
        box2_y_end = max(box2[2, 1], box2[3, 1])
        if ((box2_y_start <= box1_y_end and box2_y_start >= box1_y_start)
                or (box2_y_end <= box1_y_end and box2_y_end >= box1_y_start)
                or (box2_y_end >= box1_y_end and box2_y_start <= box1_y_start)):
            return True
        else:
            return False

    def __sort_into_lines(self, bounding_boxes):
        line_groups = []
        line_index = 0
        skipped = []
        for i in range(len(bounding_boxes)):
            if i in skipped:
                continue
            line_groups.append([])
            line_groups[line_index].append(bounding_boxes[i])
            for j in range(i + 1, len(bounding_boxes)):
                if self.__line_check(bounding_boxes[i], bounding_boxes[j]):
                    line_groups[line_index].append(bounding_boxes[j])
                    skipped.append(j)
            line_index += 1
        returnList = []
        for line in line_groups:
            line.sort(key=lambda x: min(x[0][0], x[3][0]))
            for word in line:
                returnList.append(word)
        return np.array(returnList)

    def __cut_boxes_from_image(self, image_path):
        print(f"Processing image: {image_path}")
        boxes = self.__get_bounding_boxes_for_image(image_path)
        print(f"Bounding boxes detected: {len(boxes)}")
        image = Image.open(image_path).convert("RGB")
        imagesCorners = [[int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])] for box in boxes]
        images = [image.crop((box[0], box[1], box[2], box[3])) for box in imagesCorners]
        print(f"Number of cropped boxes: {len(images)}")
        return images


    def __recongize_text(self, images):
        text_results = []
        for img in images:
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text_results.append(generated_text)
        print(f"Extracted text: {text_results}")
        return text_results


    def extract_text(self, image_path):
        images = self.__cut_boxes_from_image(image_path)
        text_results = self.__recongize_text(images)
        return text_results

model_path = "C:\\Users\\aruna\\Downloads\\H2D\\H2D\\checkpoint-2750"


pipeline = H2DPipeline(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    print(f"File uploaded to: {file_path}")
    text = pipeline.extract_text(file_path)
    return jsonify({'extracted_text': text})


if __name__ == '__main__':
    app.run(debug=True)
