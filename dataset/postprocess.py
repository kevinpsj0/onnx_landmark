import numpy as np

class Postprocessor:
    def __init__(self, norm_params):
        self.norm_params = norm_params

    def postprocess(self, model_output):
        predictions, tooth_ids = model_output
        processed_outputs = {}
        for pred, id in zip(predictions, tooth_ids):
            processed_outputs[id] = self.inverse_transform(pred, id)

        return processed_outputs
    
   
    def inverse_transform(self, landmarks, tooth_id):
        # Inverse transform labels
        processed_coords = landmarks[:, :15].reshape(-1, 3)
        axis = landmarks[:, -3:]

        # Min-Max inverse scale coordinates using norm_params
        coords = processed_coords * (self.norm_params[tooth_id]['label_maxs'] - self.norm_params[tooth_id]['label_mins']) + self.norm_params[tooth_id]['label_mins']
        
        # Combine processed data
        original_landmarks = np.concatenate([coords,axis])
        
        return original_landmarks
