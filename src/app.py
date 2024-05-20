from datetime import datetime
import joblib
import boto3
import nltk
import spacy
from nltk.corpus import stopwords

# Derval Olivera

# Parametrização 
stops = nltk.corpus.stopwords.words('portuguese')
nlp = spacy.load('pt_core_news_sm')

model = joblib.load('model.pkl')
model_vect = joblib.load('model_vect.pkl')

with open('model_version.txt', 'r', encoding='utf8') as file:
    model_version = file.read()

cloudwatch = boto3.client('cloudwatch')

def handler(event, context=False):

    print(event)
    print(context)

    labels = {
        'Hipotecas / Empréstimos': 1, 
        'Cartão de crédito / Cartão pré-pago': 2,
        'Serviços de conta bancária': 3,
        'Roubo / Relatório de disputa': 4,
        'Outros': 5
    }

    data = event
    data_procesed = prepare_payload(data)
    prediction = model.predict(data_procesed)[0]
    prediction_value = int(labels[prediction])
    text = data["data"]["text"]

    write_real_data(data, prediction)
    input_metrics(prediction_value)

    print(prediction)

    return {
        'statusCode': 200,
        'prediction': prediction,
        'version': model_version
    }


def input_metrics(prediction):
    cloudwatch.put_metric_data(
        MetricData = [
            {
                'MetricName': 'Prediction',
                'Value': prediction,
                'Dimensions': [{'Name': 'Currency', 'Value':'INR'}]
            }
        ], Namespace = 'Call Classifier Model')


def write_real_data(data, prediction):
    now = datetime.now()
    now_formatted_file = now.strftime('%Y-%m-%d')
    now_formatted = now.strftime('%d-%m-%Y %H:%M')
    
    file_name = now_formatted_file + '_classificador_data.csv'

    data['prediction'] = prediction
    data['timestamp'] = now_formatted
    data['model_version'] = model_version

    s3 = boto3.client('s3')
    bucket_name = 'fiap-jrderval'
    s3_path = 'mlops-call-classifier'

    try:
        existing_object = s3.get_object(Bucket=bucket_name, Key=f'{s3_path}/{file_name}')
        existing_data = existing_object['Body'].read().decode('utf-8').strip().split('\n')
        existing_data.append(','.join(map(str, data.values())))
        update_contet = '\n'.join(existing_data)
    except s3.exceptions.NoSuchKey:
        update_contet =','.join(data.keys()) + '\n' + ','.join(map(str, data.values()))
        

    s3.put_object(Body=update_contet, Bucket=bucket_name, Key=f'{s3_path}/{file_name}')

def prepare_payload(data):
    
    text = data["data"]["text"]
    text_lemma = lemmatizer_text(text)
    text_vect_train = model_vect.transform([text_lemma])
        
    return text_vect_train


def lemmatizer_text(text):
  sent = []
  doc = nlp(text)
  for word in doc:
      sent.append(word.lemma_)
  return " ".join(sent)