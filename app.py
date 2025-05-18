from flask import Flask, render_template, request
from model import DigitClassifier
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)

model = DigitClassifier()
model.load_state_dict(torch.load("digit_model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(), 
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
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output).item()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)
    
