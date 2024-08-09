## Environment
- Python: 3.12.4
- CUDA version: 12.1

## Installation

```
pip install .
```

## How to use
### Data Structure 
```
data/
└── 63a6ad228ea8f6001d55c084/
    ├── LowerLandmarks.json
    ├── UpperLandmarks.json
    ├── LowerTriangles.npy
    ├── UpperTriangles.npy
    ├── LowerTriangleSegment.npy
    ├── LowerVertices.npy
    └── UpperVertices.npy
└── 64b6ad228ea8f6001d55c084/
    ├── LowerLandmarks.json
    ├── UpperLandmarks.json
    ├── LowerTriangles.npy
    ├── UpperTriangles.npy
    ├── LowerTriangleSegment.npy
    ├── LowerVertices.npy
    └── UpperVertices.npy
```

- specify model location 

### For inference
```
python main.py 
```

### for visualization
```
python visualize.py
```
