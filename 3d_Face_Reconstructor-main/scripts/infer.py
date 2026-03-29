# scripts/infer.py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.network import FaceReconstructionNet
from pathlib import Path

class FaceReconstructor:
    """
    Main class for 3D face reconstruction with added normalization for robust display.
    """
    # MODIFICATION: Added a verbose flag to control printing, defaulting to off.
    def __init__(self, checkpoint_path, device=None, verbose=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = self._load_model_with_compatibility(checkpoint_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.faces = None
        self._load_3dmm_data()

    # NEW: A function to normalize the mesh for consistent visualization.
    def _normalize_vertices(self, vertices):
        """
        Centers the mesh at the origin and scales it to fit within a [-1, 1] cube.
        """
        vertices -= vertices.mean(axis=0)  # Center the mesh
        max_dist = np.max(np.linalg.norm(vertices, axis=1))
        if max_dist > 0:
            vertices /= max_dist  # Scale to fit in a unit sphere
        return vertices

    def reconstruct(self, pil_image):
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            coeffs_228, uv_preds_116k = self.model(img_tensor)

        coeffs_299 = self._convert_coefficients(coeffs_228)
        uv_preds_94k = self._convert_uv_predictions(uv_preds_116k)
        
        # Get the raw vertices
        raw_vertices = self._reconstruct_vertices(coeffs_299)
        
        # MODIFICATION: Normalize the vertices for a clean display
        normalized_vertices = self._normalize_vertices(raw_vertices)
        
        # The realistic color computation remains the same
        vertex_colors = self._compute_vertex_colors(uv_preds_94k, pil_image)

        return {
            'vertices': normalized_vertices, # Return the clean vertices
            'faces': self.faces,
            'vertex_colors': vertex_colors,
        }

    # --- All other functions below are the same, but with print() statements removed ---

    def _load_model_with_compatibility(self, checkpoint_path):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = FaceReconstructionNet(coeff_dim=228, num_triangles=116160)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_3dmm_data(self):
        data_dir = Path("data/3dmm")
        self.S_mean = np.load(data_dir / "S_mean.npy")
        self.B_id = np.load(data_dir / "B_id.npy")
        self.B_exp = np.load(data_dir / "B_exp.npy")
        self.faces = np.load(data_dir / "faces.npy")
        self.id_dims = self.B_id.shape[1]
        self.exp_dims = self.B_exp.shape[1]
        self.total_coeff_dims = self.id_dims + self.exp_dims
        self.actual_triangles = self.faces.shape[0]

    def _convert_coefficients(self, coeffs_228):
        coeffs_228_np = coeffs_228.cpu().numpy().squeeze()
        coeffs_299 = np.zeros(self.total_coeff_dims, dtype=np.float32)
        copy_length = min(len(coeffs_228_np), len(coeffs_299))
        coeffs_299[:copy_length] = coeffs_228_np[:copy_length]
        return coeffs_299

    def _convert_uv_predictions(self, uv_preds_116k):
        uv_preds_116k_np = uv_preds_116k.cpu().numpy().squeeze()
        uv_preds_94k = np.zeros((self.actual_triangles, 3, 2), dtype=np.float32)
        copy_triangles = min(uv_preds_116k_np.shape[0], uv_preds_94k.shape[0])
        uv_preds_94k[:copy_triangles] = uv_preds_116k_np[:copy_triangles]
        if copy_triangles < uv_preds_94k.shape[0]:
            avg_uv = np.mean(uv_preds_116k_np, axis=0)
            uv_preds_94k[copy_triangles:] = avg_uv
        return uv_preds_94k

    def _reconstruct_vertices(self, coeffs_299):
        id_coeff = coeffs_299[:self.id_dims]
        exp_coeff = coeffs_299[self.id_dims:self.id_dims + self.exp_dims]
        vertices = self.S_mean + self.B_id.dot(id_coeff) + self.B_exp.dot(exp_coeff)
        return vertices.reshape(-1, 3)

    def _compute_vertex_colors(self, uv_preds, pil_image):
        num_vertices = self.faces.max() + 1
        vertex_colors = np.zeros((num_vertices, 3), dtype=np.float32)
        vertex_count = np.zeros(num_vertices, dtype=np.int32)
        img_np = np.array(pil_image.convert('RGB'))
        h, w, _ = img_np.shape

        for tri_idx, tri in enumerate(self.faces):
            if tri_idx < uv_preds.shape[0]:
                uv_coords = (uv_preds[tri_idx] + 1) / 2.0
                for i, vertex_idx in enumerate(tri):
                    u, v = uv_coords[i]
                    px = int(u * (w - 1))
                    py = int((1.0 - v) * (h - 1))
                    px, py = max(0, min(px, w - 1)), max(0, min(py, h - 1))
                    vertex_colors[vertex_idx] += img_np[py, px]
                    vertex_count[vertex_idx] += 1
        
        vertex_count[vertex_count == 0] = 1
        vertex_colors /= vertex_count[:, np.newaxis]
        return np.clip(vertex_colors, 0, 255).astype(np.uint8)