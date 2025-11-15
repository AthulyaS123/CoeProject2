from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the trained model from the current directory
model = tf.keras.models.load_model("damage.keras")

IMG_HEIGHT = 128
IMG_WIDTH = 128

def preprocess_input(im_file):
    im_bytes = im_file.read()
    d = tf.io.decode_image(im_bytes, channels=3)
    d = tf.image.resize(d, (IMG_HEIGHT, IMG_WIDTH))
    d = tf.cast(d, tf.float32) / 255.0
    d = tf.expand_dims(d, 0)
    return d

@app.route("/summary", methods=["GET"])
def model_info():
    return jsonify({
        "version": "v1",
        "name": "damage_classifier_alt_lenet",
        "description": "Classifies post-Harvey satellite images as damage or no_damage.",
        "number_of_parameters": int(model.count_params())
    })

@app.route("/inference", methods=["POST"])
def classify_damage_image():
    if "image" not in request.files:
        return jsonify({"error": "user must upload an image"}), 400

    img = request.files["image"]

    try:
        data = preprocess_input(img)
    except Exception as e:
        return jsonify({"error": f"Could not process the `image` field; details: {e}"}), 400

    # Model output is sigmoid: P(label=1 = 'no_damage')
    pred = model.predict(data)
    prob_no_damage = float(pred[0][0])
    label = "no_damage" if prob_no_damage >= 0.5 else "damage"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
