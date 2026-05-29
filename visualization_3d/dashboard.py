import streamlit as st
import os
from pathlib import Path
import pandas as pd
import base64

st.set_page_config(page_title="3D Icon Inventory", page_icon="🗺️", layout="wide")

MODEL_DIR = Path("models")


@st.cache_data
def get_all_glb_files():
    categories = {}
    for category in sorted(MODEL_DIR.iterdir()):
        if category.is_dir():
            files = sorted(category.glob("**/*.glb"))
            if files:
                categories[category.name] = files
    return categories


@st.cache_data
def get_file_info(filepath):
    return os.path.getsize(filepath) / 1024


def create_viewer_html(glb_base64, filename, auto_rotate=True):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
            }}
        }}
        </script>
        <style>
            body {{ 
                margin: 0; 
                padding: 0;
                background: #0f0f1a;
                overflow: hidden;
            }}
            #info {{
                position: fixed;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 10px 15px;
                border-radius: 8px;
                font-family: Arial, sans-serif;
                font-size: 14px;
                z-index: 100;
            }}
        </style>
    </head>
    <body>
        <div id="info">Loading {filename}...</div>
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
            import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(10, 10, 10);

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            document.body.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.autoRotate = {str(auto_rotate).lower()};
            controls.autoRotateSpeed = 2;

            // Lights
            const ambient = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambient);
            
            const directional = new THREE.DirectionalLight(0xffffff, 1);
            directional.position.set(10, 20, 10);
            directional.castShadow = true;
            scene.add(directional);
            
            const point = new THREE.PointLight(0xffffff, 0.3);
            point.position.set(-10, 10, -10);
            scene.add(point);

            // Grid
            const grid = new THREE.GridHelper(40, 40, 0x444444, 0x333333);
            scene.add(grid);

            // Load GLB from base64
            const loader = new GLTFLoader();
            const blob = new Blob([Uint8Array.from(atob("{glb_base64}"), c => c.charCodeAt(0))], 
                                  {{type: 'model/gltf-binary'}});
            
            loader.load(
                URL.createObjectURL(blob),
                (gltf) => {{
                    const model = gltf.scene;
                    
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 8 / maxDim;
                    
                    model.scale.setScalar(scale);
                    model.position.sub(center.multiplyScalar(scale));
                    model.position.y = 0;
                    
                    model.traverse((child) => {{
                        if (child.isMesh) {{
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }}
                    }});
                    
                    scene.add(model);
                    camera.position.set(maxDim * 3, maxDim * 2.5, maxDim * 3);
                    controls.update();
                    
                    document.getElementById('info').textContent = 'Loaded: {filename}';
                    document.getElementById('info').style.background = 'rgba(74, 144, 217, 0.7)';
                }},
                undefined,
                (error) => {{
                    document.getElementById('info').textContent = 'Error loading model';
                    document.getElementById('info').style.background = 'rgba(220, 69, 69, 0.7)';
                }}
            );

            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html


def main():
    st.title("🗺️ 3D Icon Inventory Dashboard")

    categories = get_all_glb_files()

    if not categories:
        st.error("No GLB files found! Run `python generator.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", sum(len(f) for f in categories.values()))
    with col2:
        st.metric("Categories", len(categories))
    with col3:
        st.metric("Format", "GLB")

    st.markdown("---")

    # Model selector
    all_options = []
    for cat in sorted(categories.keys()):
        for f in categories[cat]:
            all_options.append((str(f.relative_to(MODEL_DIR)), f.name, cat))

    selected = st.selectbox(
        "🎯 Select a model:",
        options=[o[0] for o in all_options],
        format_func=lambda x: f"[{[o[2] for o in all_options if o[0] == x][0]}] {[o[1] for o in all_options if o[0] == x][0]}",
    )

    if selected:
        model_path = MODEL_DIR / selected

        with open(model_path, "rb") as f:
            glb_data = f.read()

        glb_base64 = base64.b64encode(glb_data).decode("utf-8")

        auto_rotate = st.checkbox("🔄 Auto Rotate", value=True)

        st.markdown(f"**Selected:** `{selected}` ({get_file_info(model_path):.1f} KB)")

        # Render 3D viewer
        html = create_viewer_html(glb_base64, selected, auto_rotate)

        st.components.v1.html(html, height=550, scrolling=False)

    st.markdown("---")

    # Asset table
    with st.expander("📋 View All Assets", expanded=False):
        search = st.text_input("🔍 Search:", placeholder="Filter models...")

        rows = []
        for cat in sorted(categories.keys()):
            for f in categories[cat]:
                rows.append(
                    {
                        "Category": cat,
                        "File": f.name,
                        "Path": str(f.relative_to(MODEL_DIR)),
                        "Size (KB)": f"{get_file_info(f):.1f}",
                    }
                )

        df = pd.DataFrame(rows)

        if search:
            df = df[df["File"].str.contains(search, case=False)]

        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
