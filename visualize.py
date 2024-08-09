from models.model_loader import ModelLoader
from dataset.preprocess import Preprocessor
from models.inference import Inference
from dataset.postprocess import Postprocessor
from dataset.data_loader import DataLoader
from utils.helpers import visualize, compare_predictions
import trimesh 
import numpy as np


class Pipeline:
    def __init__(self, model_path, idx = 0):
        self.model_loader = ModelLoader(model_path)
        self.inference = Inference()
        
        self.preprocessor = Preprocessor()
        self.postprocessor = Postprocessor(self.preprocessor.norm_params)
        self.dataset = DataLoader('data', idx, pretransform=self.preprocessor)

    def run(self):
        model = self.model_loader.load_model()
        raw_output = self.inference.run(model, self.dataset)
        print(raw_output[0].shape)
        final_output = self.postprocessor.postprocess(raw_output)
        return final_output

if __name__ == "__main__":
    model_path = "models/model.onnx"
    config_path = "config.yaml"
    pipeline = Pipeline(model_path, 0)
    result = pipeline.run()
    print(result)

    upper_faces, upper_vertices, upper_landmarks, _ = pipeline.dataset.get_raw('Upper')
    lower_faces, lower_vertices, lower_landmarks, _ = pipeline.dataset.get_raw('Lower')
    
    combined_vertices = np.vstack((upper_vertices, lower_vertices))
    lower_faces_adjusted = lower_faces + len(upper_vertices)
    
    # Concatenate 
    combined_labels = upper_landmarks | lower_landmarks
    combined_faces = np.vstack((upper_faces, lower_faces_adjusted))
    combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    visualize(combined_mesh, combined_labels, result)
    compare_predictions(result, combined_labels)
    