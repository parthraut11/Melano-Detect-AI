import base64
import io

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

import skin_cancer_detection as SCD

app = Flask(__name__)


CLASS_DETAILS = {
    "Actinic keratoses": {
        "name": "Actinic keratosis",
        "risk": "High priority",
        "description": (
            "A rough, sun-damaged patch that can develop into squamous cell "
            "carcinoma if it is not treated."
        ),
    },
    "Basal cell carcinoma": {
        "name": "Basal cell carcinoma",
        "risk": "Cancerous",
        "description": (
            "A common skin cancer that often appears as a pearly bump or sore, "
            "usually on sun-exposed skin."
        ),
    },
    "Benign keratosis-like lesions": {
        "name": "Benign keratosis-like lesion",
        "risk": "Usually non-cancerous",
        "description": (
            "A benign lesion that can still look suspicious in photos and should "
            "be confirmed clinically."
        ),
    },
    "Dermatofibroma": {
        "name": "Dermatofibroma",
        "risk": "Usually non-cancerous",
        "description": (
            "A firm benign skin nodule that commonly appears on the arms or legs."
        ),
    },
    "Melanocytic nevi": {
        "name": "Melanocytic nevus",
        "risk": "Usually non-cancerous",
        "description": "A common mole formed by pigment-producing skin cells.",
    },
    "Melanoma": {
        "name": "Melanoma",
        "risk": "Cancerous",
        "description": (
            "A serious form of skin cancer that needs urgent medical evaluation."
        ),
    },
    "Vascular lesions": {
        "name": "Vascular lesion",
        "risk": "Needs review",
        "description": (
            "A blood-vessel-related lesion that can sometimes bleed or mimic other "
            "skin findings."
        ),
    },
}


def load_preview_image(image_stream):
    image = Image.open(image_stream).convert("RGB")
    return image.copy()


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/platform", methods=["GET"])
def platform():
    return render_template("platform.html")


@app.route("/workflow", methods=["GET"])
def workflow():
    return render_template("workflow.html")


@app.route("/faq", methods=["GET"])
def faq():
    return render_template("faq.html")


@app.route("/showresult", methods=["POST"])
def show_result():
    uploaded_file = request.files.get("pic")
    if not uploaded_file or not uploaded_file.filename:
        return render_template(
            "home.html",
            error="Please choose an image file before submitting.",
        )

    try:
        preview_image = load_preview_image(uploaded_file.stream)
    except (UnidentifiedImageError, OSError):
        return render_template(
            "home.html",
            error="That file could not be read as an image. Please upload JPG or PNG.",
        )

    probabilities = SCD.predict(preview_image)
    top_index = int(np.argmax(probabilities))
    top_probability = float(probabilities[top_index])
    predicted_label = SCD.CLASS_NAMES[top_index]
    diagnosis = CLASS_DETAILS[predicted_label]

    ranked_predictions = []
    for index, score in sorted(
        enumerate(probabilities), key=lambda item: item[1], reverse=True
    ):
        class_label = SCD.CLASS_NAMES[index]
        class_details = CLASS_DETAILS[class_label]
        ranked_predictions.append(
            {
                "name": class_details["name"],
                "risk": class_details["risk"],
                "probability": round(float(score) * 100, 2),
            }
        )

    return render_template(
        "results.html",
        image_data=image_to_data_url(preview_image),
        diagnosis=diagnosis,
        confidence=round(top_probability * 100, 2),
        ranked_predictions=ranked_predictions,
        filename=uploaded_file.filename,
    )


if __name__ == "__main__":
    app.run(debug=True)
