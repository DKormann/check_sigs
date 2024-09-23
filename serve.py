import load_synthetic
from PIL import Image
import flask
from flask import send_file
import io

datapath = '/shared/datasets/signatures/synthetic_gpds_preprocessed_v2'
testr, testf, testl = load_synthetic.test_sample(datapath, ds_size=20, b_size=16)

print(testr.shape)

app = flask.Flask(__name__)

@app.route('/hello')
def hello(): return 'Hello, World!'

@app.route('/')
def root(): return app.send_static_file('index.html')

@app.route('/script.js')
def script(): return app.send_static_file(f'script.js')

@app.route('/style.css')
def style(): return app.send_static_file(f'style.css')


def getimage(tensor, idx):
  png0 = Image.fromarray(((1-tensor[idx])*255).astype('uint8'))
  buf = io.BytesIO()
  png0.save(buf, format='PNG')
  buf.seek(0)
  return buf

@app.route('/image/<string:col>/<int:idx>')
def image(col, idx): return send_file(getimage(testr if col =='r' else testf, idx), mimetype='image/png')

@app.route('/labels/<int:idx>')
def label(idx): return str(testl[idx].item())



def predict():
  import torch
  from model import Model 
  model = Model()
  model.load_state_dict(torch.load('model_finetuned_e_100_ test loss:   0.19  best d: 0.70 accuracy:  84.53%.pth'))
  print('model loaded')
  testr_tensor = torch.tensor(testr, dtype=torch.float32)
  testf_tensor = torch.tensor(testf, dtype=torch.float32)
  embx = model(testr_tensor)
  emby = model(testf_tensor)
  sims = torch.nn.functional.cosine_similarity(embx, emby).detach().cpu().numpy()

  # free torch memory 
  del model
  del testr_tensor
  del testf_tensor
  del embx
  del emby
  torch.cuda.empty_cache()
  return sims


predictions = predict()

print(f'predictions: {predictions}')
@app.route('/predict/<int:idx>')
def predidx(idx:int): return str(predictions[idx])



if __name__ == '__main__': app.run(host='0.0.0.0' , port=5000)