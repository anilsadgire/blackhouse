# predict.py
import json
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Replace 'your-endpoint-name' with the actual endpoint name in SageMaker
predictor = Predictor(endpoint_name='your-endpoint-name')
predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

# Input parameters for the API
request = {
    "ticker": "AAPL",
    "shares": 1000,
    "time_horizon": 390
}

# Send request to SageMaker endpoint
response = predictor.predict(json.dumps(request))
print("Trade schedule:", response)
