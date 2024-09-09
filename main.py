from models.model_loader import ModelLoader
from dataset.preprocess import Preprocessor
from models.inference import Inference
from dataset.postprocess import Postprocessor
from dataset.data_loader import DataLoader

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
    model_path = "models/v8.onnx"
    config_path = "config.yaml"
    pipeline = Pipeline(model_path, 0)
    result = pipeline.run()
    print("--------Pred-----------")
    print({key: result[key] for key in sorted(result)})

    print("------------GT---------------")
    gt = pipeline.dataset.get_raw('Upper')[2]|pipeline.dataset.get_raw('Lower')[2]
    print({key: gt[key] for key in sorted(gt)})
    
    
    