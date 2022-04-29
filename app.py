import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer 
import warnings
warnings.filterwarnings("ignore")
from sentence_splitter import SentenceSplitter
from flask import Flask, request, jsonify




app = Flask(__name__)

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
  #batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  #translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding=True,max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=1, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

@app.route('/', methods=['GET', 'POST'])
def paraphrasing():
  context = request.json['original_text']
  splitter = SentenceSplitter(language='en')
  sentence_list = splitter.split(context)
  paraphrase = []
  texto1 = ""
  pos = 0
  json_str  = []

 
  for i in sentence_list:
    a = get_response(i,1)
    paraphrase.append(a)

  for item in paraphrase:
    texto1 += paraphrase[pos][0]+" "
    pos += 1
  



  json_str.append({'original_text': context,
                  'new_text': texto1})

  return jsonify(json_str)



if __name__ == '__main__':
    app.run()


