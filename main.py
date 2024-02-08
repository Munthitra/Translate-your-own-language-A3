import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
from torchtext.data.utils import get_tokenizer
from models import *  # Import your model definition
# from utils import collate_batch  # Import your data processing utilities

input_dim   = 6327
output_dim  = 6545
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1
additive_attention = AdditiveAttention(head_dim=hid_dim // enc_heads)

SRC_PAD_IDX = 1
TRG_PAD_IDX = 1

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_transform = torch.load('./models/vocab')['en']  # For tokenizing english text to numerical tokens
mapping = torch.load('./models/vocab')['th'].get_itos()  # For transforming numerical output to Thai text
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

enc = Encoder(input_dim,
              hid_dim,
              enc_layers,
              enc_heads,
              enc_pf_dim,
              enc_dropout,
              device,
              additive_attention,
              max_length=1000
              # max_length=len(mapping),
              )

dec = Decoder(output_dim,
              hid_dim,
              dec_layers,
              dec_heads,
              dec_pf_dim,
              enc_dropout,
              device,
              additive_attention,
              max_length=1000
              # max_length=len(mapping),
              )

model_path = './models/Seq2SeqTransformer.pt'
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Machine Translation Demo"),
    dcc.Textarea(id='input-text', placeholder="Enter text in source language...", rows=4, cols=50),
    html.Button('Translate', id='translate-button', n_clicks=0),
    html.Div(id='output-text')
])

# Define translation function
def translate_text(input_text):
    # Preprocess input text (tokenization, etc.)
    # Example: tokenized_input = preprocess_input(input_text)

    # Perform translation using the model
    # Example: translated_output = model.translate(tokenized_input)

    # Return translated output
    # Example: return translated_output
    pass

# Define callback to translate text when button is clicked
@app.callback(
    Output('output-text', 'children'),
    [Input('translate-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if n_clicks > 0:
        tokenized_prompt = ['<sos>'] + tokenizer(input_text) + ['<eos>']  # tokenize then concatenate special tags to the start and end of list
        num_tokens = vocab_transform(tokenized_prompt)  # convert to numerical representations
        model_input = torch.tensor(num_tokens, dtype=torch.int64).reshape(1, -1).to(device)  # prepare model input
        model_output = generate(model, model_input)[0]
        print(model_output)
        translated_text = ''.join([mapping[token.item()] for token in model_output])
        # translated_text = translate_text(input_text)
        return translated_text

if __name__ == '__main__':
    app.run_server(debug=True)

