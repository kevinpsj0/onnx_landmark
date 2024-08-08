import onnxruntime as ort

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        session = ort.InferenceSession(self.model_path)
        return session