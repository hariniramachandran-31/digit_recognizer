from flask import Flask, render_template, request
from PIL import Image
import torch
import os

# Load your model and transforms (make sure you define them)
# Example placeholders:
# model = torch.load('model.pth', map_location=torch.device('cpu'))
# model.eval()
# transform = transforms.Compose([...])

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        image = request.files["digit"]
        img = Image.open(image)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output).item()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    # This makes the app compatible with Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))