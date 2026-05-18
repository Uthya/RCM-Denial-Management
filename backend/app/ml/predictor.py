from app.ml.model_loader import load_model


class DenialPredictor:
    def __init__(self):
        self.model = None

    def load(self):
        self.model = load_model()

    def predict(self, features: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # Placeholder for prediction logic
        return {"denial_probability": 0.0, "risk_level": "low"}
