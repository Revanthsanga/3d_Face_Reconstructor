# scripts/mesh_utils.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def render_mesh(faces, vertices, vertex_colors, texture=None, mode="textured", img_size=512):
    """
    Enhanced mesh rendering with multiple visualization modes
    
    Args:
        faces: (num_triangles, 3) face indices
        vertices: (num_vertices, 3) 3D vertex coordinates  
        vertex_colors: (num_vertices, 3) RGB colors per vertex
        texture: HxWx3 texture image (optional)
        mode: "wireframe", "textured", "pointcloud", or "shaded"
        img_size: output image size
    
    Returns:
        Rendered image as numpy array
    """
    
    if mode == "wireframe":
        return render_wireframe(faces, vertices, vertex_colors, img_size)
    elif mode == "textured":
        return render_textured(faces, vertices, vertex_colors, texture, img_size)
    elif mode == "pointcloud":
        return render_pointcloud(vertices, vertex_colors, img_size)
    elif mode == "shaded":
        return render_shaded(faces, vertices, vertex_colors, img_size)
    else:
        return render_textured(faces, vertices, vertex_colors, texture, img_size)

def render_wireframe(faces, vertices, vertex_colors, img_size=512):
    """Render 3D mesh as wireframe with vertex colors"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Project vertices to 2D
    projected_vertices = project_vertices(vertices, img_size)
    
    # Draw edges
    for face in faces:
        for i in range(3):
            start_idx = face[i]
            end_idx = face[(i + 1) % 3]
            
            start_pos = projected_vertices[start_idx]
            end_pos = projected_vertices[end_idx]
            
            # Get color from starting vertex
            color = tuple(map(int, vertex_colors[start_idx]))
            
            # Draw line
            cv2.line(img, 
                    (int(start_pos[0]), int(start_pos[1])),
                    (int(end_pos[0]), int(end_pos[1])),
                    color, 2, cv2.LINE_AA)
    
    return img

def render_textured(faces, vertices, vertex_colors, texture, img_size=512):
    """Render textured 3D face with smooth coloring"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Project vertices to 2D
    projected_vertices = project_vertices(vertices, img_size)
    
    # Draw filled triangles with vertex colors
    for face in faces:
        pts = projected_vertices[face].astype(np.int32)
        
        if len(pts) == 3:  # Ensure we have a valid triangle
            # Create mask for the triangle
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)
            
            # Get vertex colors for this face
            face_colors = vertex_colors[face]
            
            # Create smooth color gradient within triangle
            if texture is not None:
                # Use texture if available
                try:
                    # Simple texture mapping (placeholder - can be enhanced)
                    tex_coords = projected_vertices[face] / img_size
                    tex_coords = np.clip(tex_coords, 0, 1)
                    h, w = texture.shape[:2]
                    tex_pts = (tex_coords * [w-1, h-1]).astype(int)
                    avg_color = np.mean(texture[tex_pts[:,1], tex_pts[:,0]], axis=0) * 255
                except:
                    avg_color = np.mean(face_colors, axis=0)
            else:
                # Use vertex colors
                avg_color = np.mean(face_colors, axis=0)
            
            # Apply color to the triangle
            colored_triangle = np.zeros_like(img)
            colored_triangle[mask > 0] = avg_color
            img = np.where(mask[:,:,None] > 0, colored_triangle, img)
    
    return img

def render_pointcloud(vertices, vertex_colors, img_size=512):
    """Render 3D point cloud"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Project vertices to 2D
    projected_vertices = project_vertices(vertices, img_size)
    
    # Draw points
    for i, (point, color) in enumerate(zip(projected_vertices, vertex_colors)):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img_size and 0 <= y < img_size:
            cv2.circle(img, (x, y), 2, color.tolist(), -1)
    
    return img

def render_shaded(faces, vertices, vertex_colors, img_size=512):
    """Render with simple shading based on face normals"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Project vertices to 2D
    projected_vertices = project_vertices(vertices, img_size)
    
    # Calculate face normals for shading
    face_normals = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        face_normals.append(normal)
    
    face_normals = np.array(face_normals)
    
    # Simple shading based on Z-component of normal (facing camera)
    shading = (face_normals[:, 2] + 1) / 2  # Convert to [0,1]
    
    # Draw shaded faces
    for i, face in enumerate(faces):
        pts = projected_vertices[face].astype(np.int32)
        
        if len(pts) == 3:
            # Get base color from vertices
            base_color = np.mean(vertex_colors[face], axis=0)
            
            # Apply shading
            shaded_color = (base_color * shading[i]).astype(int)
            shaded_color = np.clip(shaded_color, 0, 255)
            
            # Draw triangle
            cv2.fillConvexPoly(img, pts, shaded_color.tolist())
    
    return img

def project_vertices(vertices, img_size):
    """Project 3D vertices to 2D screen coordinates with better centering"""
    # Center and scale vertices
    centered = vertices - vertices.mean(axis=0)
    
    # Simple orthographic projection (front view - use X and Y)
    projected = centered[:, :2]
    
    # Normalize to fit in image
    max_extent = np.max(np.abs(projected))
    if max_extent > 0:
        scale = (img_size * 0.45) / max_extent  # Leave some border
        projected = projected * scale
    
    # Center in image
    projected += img_size // 2
    
    return projected

def save_3d_model(vertices, faces, vertex_colors, filename):
    """
    Save 3D model as OBJ file with vertex colors as comments
    
    Args:
        vertices: (N, 3) vertex coordinates
        faces: (M, 3) face indices  
        vertex_colors: (N, 3) RGB colors
        filename: output file path
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("# 3D Face Reconstruction Model\n")
        f.write("# Generated by AI Face Reconstruction System\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        
        # Write vertices with colors as comments
        for i, (v, color) in enumerate(zip(vertices, vertex_colors)):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write(f"# Color {color[0]} {color[1]} {color[2]}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        f.write(f"\n# Model complete\n")

def create_texture_from_uvs(uv_preds, size=256):
    """
    Create a simple procedural texture based on predicted UV coordinates
    """
    texture = np.ones((size, size, 3), dtype=np.float32) * 0.8  # Light gray base
    
    # Add some procedural patterns based on UV distribution
    for u in range(size):
        for v in range(size):
            # Simple gradient based on position
            u_norm = u / size
            v_norm = v / size
            
            # Add some skin-like color variations
            r = 0.6 + 0.2 * np.sin(u_norm * np.pi * 2) * 0.1
            g = 0.5 + 0.2 * np.cos(v_norm * np.pi * 2) * 0.1  
            b = 0.4 + 0.1 * np.sin((u_norm + v_norm) * np.pi) * 0.1
            
            texture[v, u] = [r, g, b]
    
    return np.clip(texture, 0, 1)