from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from model_utils import load_meta_model, predict_comment
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_models, meta_model = load_meta_model(device)
tokenizer = torch.load('./models/bert_tokenizer.pth')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    comment = data.get('comment', '')

    predicted_label, predicted_class_name = predict_comment(
        comment=comment,
        base_models=base_models,
        meta_model=meta_model,
        tokenizer=tokenizer,
        device=device
    )

    return jsonify({
        'label': predicted_label,
        'class_name': predicted_class_name
    })

if __name__ == '__main__':
    app.run(debug=True)
