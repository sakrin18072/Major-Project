import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        if not isinstance(comment, str):
            print(f"Non-string comment at index {idx}: {comment}")
        label = self.labels[idx]
        encoding = self.tokenizer(
            comment,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertTransformer(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.3):
        super(BertTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, 
                                                   nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state.transpose(0, 1) 
        transformer_output = self.transformer_encoder(hidden_states)  
        cls_output = transformer_output[0, :, :] 
        return self.fc(cls_output)

class BertGRU(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BertGRU, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 384), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state 
        gru_output, _ = self.gru(hidden_states) 
        cls_output = gru_output[:, 0, :] 
        return self.fc(cls_output)

class BertRNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.rnn = nn.RNN(self.bert.config.hidden_size, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 384),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state 
        rnn_output, _ = self.rnn(hidden_states)  
        cls_output = rnn_output[:, 0, :] 
        return self.fc(cls_output)

class BertLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=2, dropout=0.3):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 384), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state 
        lstm_output, _ = self.lstm(hidden_states) 
        cls_output = lstm_output[:, 0, :] 
        return self.fc(cls_output)

class BertFFN(nn.Module):
    def __init__(self, num_classes):
        super(BertFFN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)
    
class MetaModel(nn.Module):
    def __init__(self, num_classes, num_models, model_output_size):
        super(MetaModel, self).__init__()
        input_size = num_models * model_output_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, meta_features):
        return self.fc(meta_features)

def load_model_ffn(model_path, num_classes=3):
    model = BertFFN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_rnn(model_path, num_classes=3):
    model = BertRNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_lstm(model_path, num_classes=3):
    model = BertLSTM(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_gru(model_path, num_classes=3):
    model = BertGRU(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_te(model_path, num_classes=3):
    model = BertTransformer(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
def load_meta_model(device):
    
    base_models = [
    load_model_ffn('./models/bert_ffn_model.pth').to(device),
    load_model_gru('./models/bert_gru_model.pth').to(device),
    load_model_lstm('./models/bert_lstm_model.pth').to(device),
    load_model_te('./models/bert_transformer_model.pth').to(device),
    load_model_rnn('./models/bert_rnn_model.pth').to(device)
    ]

    meta_model = MetaModel(3,5,3)
    meta_model.load_state_dict(torch.load('./models/bert_meta_model.pth'))
    meta_model.to(device)
    meta_model.eval()  

    return base_models, meta_model

def predict_comment(comment, base_models, meta_model, tokenizer, device):
    encoded_input = tokenizer(
        comment,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    meta_features = []
    for model in base_models:
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask) 
        meta_features.append(outputs)

    meta_features = torch.cat(meta_features, dim=1)  

    meta_model.eval()
    with torch.no_grad():
        meta_outputs = meta_model(meta_features)
        predicted_class = torch.argmax(meta_outputs, dim=1).item()

    class_names = ['Not Abusive', 'Hate Speech', 'Hate + Abusive']

    return predicted_class, class_names[predicted_class]