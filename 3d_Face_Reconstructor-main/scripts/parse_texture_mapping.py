import json
import argparse
import os

def parse_json_to_tri_uvs(json_p, faces_p, out_p):
    """
    Parse texture mapping JSON to generate per-triangle UVs.
    Automatically converts vertex-based UVs if tri_uvs is missing.
    """
    # Load the JSON file
    with open(json_p, "r") as f:
        json_data = json.load(f)

    # Attempt to get tri_uvs directly
    if "tri_uvs" in json_data:
        tri_uvs = json_data["tri_uvs"]
        print(f"[Info] Found tri_uvs in JSON, total triangles: {len(tri_uvs)}")
    else:
        # Attempt to generate tri_uvs from vertex-based UVs
        texture_mapping = json_data.get("textureMapping", {})
        point_data = texture_mapping.get("pointData", [])
        triangles = texture_mapping.get("triangles", [])

        if not point_data or not triangles:
            raise RuntimeError(
                "No per-triangle UV candidate found and cannot generate from vertex data!"
            )

        # Generate tri_uvs
        tri_uvs = []
        for tri in triangles:
            try:
                uv0 = point_data[tri[0]]
                uv1 = point_data[tri[1]]
                uv2 = point_data[tri[2]]
            except IndexError:
                raise RuntimeError(f"Invalid triangle indices {tri} for pointData of length {len(point_data)}")
            tri_uvs.append([uv0, uv1, uv2])

        print(f"[Info] Generated tri_uvs from vertex UVs, total triangles: {len(tri_uvs)}")

    # Optional: validate faces file if needed (existing code)
    if not os.path.exists(faces_p):
        print(f"[Warning] Faces file {faces_p} does not exist. Skipping validation.")
    else:
        # Here you can load faces and validate if required
        pass

    # Save tri_uvs JSON to output path
    with open(out_p, "w") as f:
        json.dump({"tri_uvs": tri_uvs}, f, indent=2)

    print(f"[Success] tri_uvs JSON saved to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse texture mapping to tri_uvs.")
    parser.add_argument("--json", required=True, help="Path to input JSON with texture mapping")
    parser.add_argument("--faces", required=True, help="Path to faces file (can skip validation)")
    parser.add_argument("--out", required=True, help="Path to output tri_uvs JSON")
    args = parser.parse_args()

    parse_json_to_tri_uvs(args.json, args.faces, args.out)
