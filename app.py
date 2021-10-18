import pandas as pd
import transformers
transformers.logging.set_verbosity_error()

from preprocess_data import preprocess_data
from model import create_model
from predict_output import predict
from get_metric import get_metric
from tokenizer import get_tokenizer

from fastapi import FastAPI
import uvicorn


app = FastAPI(title = "Toxicity classification model",
              description="""This is a toxicity classification model based on Logistic regression"
                          and LightGBM algorithms""")

tags_metadata = [
    {
        "name": "Toxicity prediction"
    }]


def get_predictions(input):
  tokenizer = get_tokenizer('./archive (2)/roberta_tokenizer')

  input_ids,attention_mask,input = preprocess_data(input,128,tokenizer)
  
  model = create_model(128,0.1,'./archive (1)/mymodel_pretrained','./archive/checkpt1/model_roberta.hdf5')

  input_data = (input_ids,attention_mask)
  pred_text= predict(model,input_data,tokenizer,input)
  return pred_text

def get_prediction_by_text(text : str, sentiment : str):
    # vals = [each for each in text]
    values = {'text': [text], 'sentiment': [sentiment]}
    df = pd.DataFrame(values)
    output = get_predictions(df)
    return output
    # return render_template('final.html', text=vals[0], prediction=output[0], sentiment=vals[1])


@app.get("/get-prediction", response_model= int, tags=['Toxicity prediction'])
def result(text : str, sentiment: str):
    result = get_prediction_by_text(text=text, sentiment=sentiment)

    return result



if __name__ == '__main__':
    uvicorn.run(app)
