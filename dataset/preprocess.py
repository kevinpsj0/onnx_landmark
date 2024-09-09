import numpy as np
import trimesh
import pyfqmr

class Preprocessor:
    def __init__(self, num_paddding = 0, num_points = 8000):
        self.num_padding = num_paddding
        self.num_points = num_points
        self.norm_params = {}

    def __call__(self, input_data) : 
        faces, vertices, landmarks, segmentation = input_data
        return self.preprocess(faces, vertices, landmarks, segmentation)

    def preprocess(self, faces, vertices, landmarks, segmentation):
        processed_inputs = []
        # Crop
        faces, vertices, landmarks, bbs, tooth_ids  = self.crop_bounding_box(faces, vertices, landmarks, segmentation, padding = self.num_padding)
        for face, vertice, landmark, bb, tooth_id in zip(faces, vertices, landmarks, bbs, tooth_ids) :      
            # Downscale
            f, v = self.downscale_mesh(vertices = vertice, faces = face, num_points = self.num_points)

            # Compute Centroids
            centroids = self.compute_centroids(f, v)
            
            # Normalize
            self.fit(v, bb, tooth_id)
            centroids, landmark = self.transform(centroids, landmark, tooth_id)
            processed_inputs.append((centroids,landmark, tooth_id))
        
        return processed_inputs

    def crop_bounding_box(self, faces, vertices, landmarks, segmentation, padding = 0) : 
        """
        Crop out a bounding box around the segmented object in the mesh, ensuring all coordinates are included
        and the bounding box is padded to each direction.

        Parameters:
        - mesh: The mesh object containing vertices and faces.
        - segmentation: A numpy array where each element is the segmentation label of a face.
        - labels: The label that has Tooth_id, 'CR', 'DistalRef', 'Face', 'MesialRef', 'TipRef', 'ToothAxis' coordinate information.
        - padding: Additional padding to apply in all directions.

        Returns:
        - cropped_vertices: List of cropped vertices containing the segmented objects with padding.
        - cropped_triangles: List of cropped triangles containing the segmented objects with padding.
        - cropped_labels: List of labels corresponding to the cropped_meshes.
        """
        cropped_vertices = []
        cropped_triangles = []
        cropped_labels = []
        bounding_boxes = []
        tooth_ids = []

        for tooth_id, cropped_label in landmarks.items():
            # Identify faces belonging to the specified segment
            
            segment_faces = faces[segmentation == int(tooth_id)]

            # Identify vertices belonging to these faces
            segment_vertices_indices = np.unique(segment_faces)
            segment_vertices = vertices[segment_vertices_indices]
            
            # Calculate the bounding box
            min_coords = np.min(segment_vertices, axis=0)
            max_coords = np.max(segment_vertices, axis=0)
            
            # Apply specific padding to bounding box
            min_coords[0] -= 4 + padding
            max_coords[0] += 4 + padding
            min_coords[1] -= 4 + padding
            max_coords[1] += 4 + padding
            min_coords[2] -= 7 + padding
            max_coords[2] += 7 + padding
            
            # Crop the mesh
            # Identify all vertices within the padded bounding box
            within_bbox = np.all((vertices >= min_coords) & (vertices <= max_coords), axis=1)
            cropped_vertice = vertices[within_bbox]
            
            # Identify faces that are entirely within the bounding box
            face_mask = np.all(within_bbox[faces], axis=1)
            cropped_face = faces[face_mask]
            
            # Update face indices to match the new vertex array
            old_to_new_indices = -np.ones(vertices.shape[0], dtype=int)
            old_to_new_indices[within_bbox] = np.arange(np.sum(within_bbox))
            cropped_face = old_to_new_indices[cropped_face]
            
            
            cropped_triangles.append(cropped_face)
            cropped_vertices.append(cropped_vertice)
            cropped_labels.append(cropped_label)
            bounding_boxes.append((min_coords, max_coords))
            tooth_ids.append(tooth_id)
        return cropped_triangles, cropped_vertices, cropped_labels, bounding_boxes, tooth_ids 
    
    def compute_centroids(self, faces, vertices) : 
        input = vertices[faces]
        input = input.mean(axis=1)
        return input

    def fit(self, vertices, bounding_box, tooth_id) :
        self.norm_params[tooth_id] = {}
        self.norm_params[tooth_id]['label_mins'] = bounding_box[0]
        self.norm_params[tooth_id]['label_maxs'] = bounding_box[1]
        self.norm_params[tooth_id]['means'] = vertices.mean(axis=0)
        self.norm_params[tooth_id]['stds'] = vertices.std(axis=0)
        self.norm_params[tooth_id]['mins'] = vertices.min(axis=0)
        self.norm_params[tooth_id]['maxs'] = vertices.max(axis=0)

    def transform(self, centroids, landmarks, tooth_id) : 
        # Transform centroids
        landmarks = landmarks.copy()
        for i in range(3) : 
            centroids[:, i] = (centroids[:, i] - self.norm_params[tooth_id]['mins'][i]) / (self.norm_params[tooth_id]['maxs'][i] - self.norm_params[tooth_id]['mins'][i])  

        # Transform labels
        coords = landmarks[0:5]
        axis = landmarks[-1]

        # Min-Max scale coordinates using norm_params
        processed_coords = (coords - self.norm_params[tooth_id]['label_mins']) / (self.norm_params[tooth_id]['label_maxs'] - self.norm_params[tooth_id]['label_mins'])
        processed_coords = processed_coords.reshape(-1, 15)  # Reshape to 15 xyzs for each row
        
        # Normalize tooth axis to have magnitude 1
        normalized_axis = axis / np.linalg.norm(axis, axis=0, keepdims=True)
        
        # Combine processed data
        processed_landmarks = np.hstack((processed_coords, normalized_axis.reshape(-1,3)))
        
        return np.array(centroids).astype(np.float32), np.array(processed_landmarks).astype(np.float32)

    @staticmethod
    def downscale_mesh(faces, vertices, num_points: int = 8000) -> trimesh.Trimesh:
        mesh_simplifier = pyfqmr.Simplify()
        # Load mesh 
        mesh_simplifier.setMesh(vertices, faces)
        mesh_simplifier.simplify_mesh(target_count=num_points, aggressiveness=3, preserve_border=True, verbose=0,max_iterations=2000)
        new_positions, new_face, _ = mesh_simplifier.getMesh()
        mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
        
        # Simplified mesh
        vertices = mesh_simple.vertices
        faces = mesh_simple.faces
        if faces.shape[0] < num_points: 
            fs_diff = num_points - faces.shape[0]
            faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
        elif faces.shape[0] > num_points:
            mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
            samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, num_points)
            mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
            faces = mesh_simple.faces
            vertices = mesh_simple.vertices
        return faces, vertices


