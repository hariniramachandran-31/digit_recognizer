from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the trained model
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        image = request.files["digit"]
        img = Image.open(image)
        img = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output).item()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)