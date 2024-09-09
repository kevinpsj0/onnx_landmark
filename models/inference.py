import numpy as np 
class Inference:
    def __init__(self):
        pass

    def run(self, model, dataloader):
        all_predictions = []
        tooth_ids = []
        for inputs, gt, id, in dataloader:
                        
            # Run ONNX model inference
            inputs = np.expand_dims(inputs, axis = 0)
            id = np.expand_dims(id, axis = 0).astype(np.int64)
            ort_inputs = {'l_xyz_': inputs, 'l_class_ids_': id}
            ort_outs = model.run(None, ort_inputs)
            
            # Collect predictions
            all_predictions.append(ort_outs[0])
            tooth_ids.append(id)
        # Concatenate predictions for all batches
        return (np.array(all_predictions).squeeze(2), tooth_ids)