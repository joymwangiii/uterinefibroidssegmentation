from flask import Flask, render_template, request
import onnxruntime
import base64
import numpy as np
import cv2
import io

app = Flask(__name__)

# Load the ONNX model
onnx_model_path = '134177_Model(4).onnx'  # Update with the path to your ONNX model
sess = onnxruntime.InferenceSession(onnx_model_path)

# Print input details
input_details = sess.get_inputs()
for input_detail in input_details:
    print(f"Input Name: {input_detail.name}, Shape: {input_detail.shape}, Type: {input_detail.type}")

output_details = sess.get_outputs()
print("\nOutput Names:")
for output_detail in output_details:
    print(f"  {output_detail.name}, Shape: {output_detail.shape}, Type: {output_detail.type}")


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, ...].astype(np.float32) / 255.0
    return image

def predict_segmentation_with_ellipses(img_path, threshold=0.5):
    input_data = preprocess_image(img_path)

    if input_data.shape != (1, 3, 256, 256):
        input_data = input_data.reshape((1, 3, 256, 256))

    input_name = 'input.1'
    output_names = ['523', '537']

    output = sess.run(None, {input_name: input_data})

    segmentation_probs = output[1][0, 1, :, :]
    segmentation_mask = (segmentation_probs > threshold).astype(np.uint8) 


    segmentation_mask = (segmentation_mask * 255).astype(np.uint8)

    print("Segmentation Mask Shape:", segmentation_mask.shape)
    print("Unique Values in Segmentation Mask:", np.unique(segmentation_mask))


    # Find contours in the binary mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the contours
    contour_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Draw ellipses for contours with enough points
    original_image = cv2.imread(img_path)
    contour_overlay = original_image.copy()
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse = (ellipse[0], (ellipse[1][0] * 0.9, ellipse[1][1] * 0.9), ellipse[2])
            cv2.ellipse(contour_overlay, ellipse, (0, 255, 0), 2)  # Draw ellipses in green

    return segmentation_mask, contour_overlay

@app.route("/", methods=['GET', 'POST'])
def homepage():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    img_path = None
    overlay_image_base64 = None

    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        segmentation_mask, contour_overlay = predict_segmentation_with_ellipses(img_path)

        if not contour_overlay is None:
            # Convert contour_overlay to base64 for rendering in HTML
            _, buffer_overlay = cv2.imencode('.jpg', contour_overlay)
            overlay_image_base64 = base64.b64encode(buffer_overlay).decode('utf-8')

    return render_template("index.html", img_path=img_path, overlay_image_base64=overlay_image_base64)

if __name__ == "__main__":
    app.run(debug=True)



