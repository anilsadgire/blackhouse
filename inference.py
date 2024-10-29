# inference.py
from stable_baselines3 import SAC

def model_fn(model_dir):
    """Load the model from the model directory for deployment."""
    return SAC.load(f"{model_dir}/rl_model_sagemaker.zip")

def predict_fn(input_data, model):
    """Predict using the model with the provided input data."""
    import json
    data = json.loads(input_data)
    action, _states = model.predict([data['shares'], data['time_horizon']])
    return {'action': action.tolist()}  # Convert to JSON serializable format
