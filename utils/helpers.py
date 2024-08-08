
import os 
import json
import numpy as np
import plotly.graph_objects as go

coordinate_names = ['CR', 'DistalRef', 'Face', 'MesialRef', 'TipRef', 'ToothAxis']

def data_exists(case_path, pre):
    """
    Checks if all necessary data files exist for a given case and prefix.

    Args:
        case_path (str): Path to the case directory.
        pre (str): Prefix indicating either "Upper" or "Lower" jaw.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    triangles_path = os.path.join(case_path, pre + 'Triangles.npy')
    trianglesegments_path = os.path.join(case_path, pre + 'TriangleSegment.npy')
    vertices_path = os.path.join(case_path, pre + 'Vertices.npy')
    landmarks_path = os.path.join(case_path, pre + 'Landmarks.json')
    
    return os.path.exists(triangles_path) and os.path.exists(trianglesegments_path) and os.path.exists(vertices_path) and os.path.exists(landmarks_path)

def load_landmarks(landmarks_path):
    """
    Loads landmarks from a JSON file into a list of tuples.

    Args:
        landmarks_path (str): Path to the landmarks JSON file.

    Returns:
        list: List of tuples containing (tooth ID, NumPy array of coordinates)
    """

    with open(landmarks_path, 'r') as f:
        data = json.load(f)
    
    data_list = {}

    for tooth_id, tooth_data in data.items():
        coordinates = []
        for coord_name in coordinate_names:
            coords = tooth_data.get(coord_name, [0.0, 0.0, 0.0])
            coordinates.append(np.array(coords))

        data_list[tooth_id] = np.array(coordinates)
    return data_list

def import_data(case_path, pre="Upper"):
    """
    Imports mesh data and landmarks.

    Args:
        case_path (str): Path to the case directory.
        pre (str): Prefix indicating either "Upper" or "Lower" jaw.

    Returns:
        tuple: Contains mesh, landmarks, and triangle segments if all files exist, otherwise None.
    """
    triangles_path = os.path.join(case_path, pre + 'Triangles.npy')
    trianglesegments_path = os.path.join(case_path, pre + 'TriangleSegment.npy')
    vertices_path = os.path.join(case_path, pre + 'Vertices.npy')
    landmarks_path = os.path.join(case_path, pre + 'Landmarks.json')
    
    if data_exists(case_path, pre):
        triangles = np.load(triangles_path)
        trianglesegments = np.load(trianglesegments_path)
        vertices = np.load(vertices_path)
        landmarks = load_landmarks(landmarks_path)
        return triangles, vertices, landmarks, trianglesegments
    else:
        return None


def visualize(mesh=None, gt_label=None, pred_label=None, file_name = None):
    traces = []
    title = file_name if file_name else "Mesh"
    # Add mesh trace if available
    if mesh is not None:
        traces.append(create_mesh(mesh))

    # Colors for different coordinate types
    colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']

    # Visualize label data
    if gt_label is not None:
        traces += coord_to_trace(gt_label, "GT", colors, 'circle')
    
    # Visualize predicted labels if available
    if pred_label is not None:
        traces += coord_to_trace(pred_label, "Pred", colors, 'diamond')
    
    # Create figure and add traces
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=title,
        legend=dict(itemsizing='constant')
    )
    fig.show()

def coord_to_trace(dict_label, legendgroup, colors, symbol) : 
    traces = []
    for id, coords in dict_label.items():
        for j, name in enumerate(coordinate_names):
            if name == "ToothAxis" : 
                center = coords[0]
                direction = coords[-1]
                start = center - direction * 1
                end = center + direction * 1
                trace = create_arrow(
                        start, end, colors[j], f'{name} {id}', legendgroup
                    )
            else:
                trace = create_point_trace(
                    coords[j],
                    color = colors[j], name = f'{legendgroup} {name} {id}', symbol = symbol, legendgroup= legendgroup
                )
            traces.append(trace)
    return traces

def create_arrow(start, end, color, name, legendgroup, showlegend=True):
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=5),
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend
    )

def create_point_trace(xyz, color, name, symbol, legendgroup, showlegend=True):
    return go.Scatter3d(
        x=[xyz[0]], y=[xyz[1]], z=[xyz[2]],
        mode='markers',
        marker=dict(size=5, color=color, symbol = symbol, opacity=1),
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend
    )

def create_mesh(mesh, color = 'lightgray', title = 'Mesh') :
    vertices = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color= color,
            opacity=0.6,
            name = title
        )

def compare_predictions(predictions, ground_truth):
    keys = sorted(predictions.keys())  
    
    for key in keys:
        pred = predictions[key]
        gt = ground_truth[key]
        
        print(f"Comparison for key {key}:")
        print("Prediction:\t\t\tGround Truth:")
        print("-" * 80)
        for p, g in zip(pred, gt):
            print(f"{p}\t\t{g}")
        print("\n" + "=" * 80 + "\n")
