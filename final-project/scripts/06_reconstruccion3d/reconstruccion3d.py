#!/usr/bin/env python3
"""
Módulo 6 — RECONSTRUCCIÓN 3D / MALLAS + VISOR (Three.js)
=======================================================
Genera mallas 3D del Tumor Realzante (ET) para comparar PREDICCIÓN (método estrella
RegionGrowing) vs GROUND TRUTH, y un visor web interactivo autocontenido (Three.js local).

Casos: los 8 CON ET de output/segmentacion/outputs/casos_segmentacion.json.

Por caso, dos mallas (marching cubes con spacing físico real + suavizado ligero):
  * ET predicho  : output/segmentacion/outputs/<caso>-regiongrowing-ET.nii.gz
  * ET GT (seg==3): leído del ZIP con comun/io_zip + LABEL_ET.

Produce (en output/reconstruccion3d/):
  outputs/<caso>-ET-pred.obj  / .glb   y  <caso>-ET-gt.obj / .glb
  outputs/metricas_3d.csv
  figuras/render_<caso>.png            (render estático matplotlib 3D)
  lib/                                 (Three.js descargado localmente)
  visor_3d.html                        (visor interactivo autocontenido, GLB embebidos)
  reconstruccion3d_reporte.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C          # noqa: E402
from comun import io_zip                    # noqa: E402
from comun import reporte as R              # noqa: E402

import matplotlib.pyplot as plt             # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

MIN_VOX = 20            # máscaras más pequeñas se omiten (malla degenerada)
SMOOTH_SIGMA = 0.6      # suavizado gaussiano de la máscara antes de marching cubes
COLOR_PRED = "#ff7f0e"  # naranja (predicción)
COLOR_GT = "#2ca02c"    # verde (ground truth)


# --------------------------------------------------------------------------- #
# Malla a partir de una máscara binaria
# --------------------------------------------------------------------------- #
def malla_de_mascara(mask: np.ndarray, spacing_xyz):
    """
    Máscara binaria (z,y,x) -> trimesh.Trimesh en mm (spacing aplicado), o None si es
    demasiado pequeña. Suaviza la máscara y cierra la superficie (padding) para que la
    malla no se vea escalonada y el volumen sea válido.
    """
    if mask.sum() < MIN_VOX:
        return None
    m = np.pad(mask.astype(np.float32), 1)                  # cierra superficie en bordes
    m = ndi.gaussian_filter(m, sigma=SMOOTH_SIGMA)
    if m.max() < 0.5:
        return None
    # spacing en orden de ejes del array (z, y, x): spacing_xyz = (sx, sy, sz)
    sx, sy, sz = spacing_xyz
    verts, faces, normals, _ = marching_cubes(m, level=0.5, spacing=(sz, sy, sx))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    trimesh.smoothing.filter_taubin(mesh, iterations=8)     # suavizado Taubin (no encoge)
    mesh.fix_normals()
    return mesh


def distancias_superficie(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh):
    """Distancia media simétrica y Hausdorff (mm) entre los vértices de dos mallas."""
    ta, tb = cKDTree(mesh_a.vertices), cKDTree(mesh_b.vertices)
    da, _ = tb.query(mesh_a.vertices)     # a -> b
    db, _ = ta.query(mesh_b.vertices)     # b -> a
    media = float(np.mean(np.concatenate([da, db])))
    hausdorff = float(max(da.max(), db.max()))
    return round(media, 3), round(hausdorff, 3)


# --------------------------------------------------------------------------- #
# Render estático (matplotlib 3D) para el reporte
# --------------------------------------------------------------------------- #
def render_estatico(cid, mesh_pred, mesh_gt, figuras: Path, max_caras=3500):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0e1116")
    for mesh, color, alpha in [(mesh_gt, COLOR_GT, 0.35), (mesh_pred, COLOR_PRED, 0.9)]:
        if mesh is None:
            continue
        faces = mesh.faces
        if len(faces) > max_caras:                          # decima solo para visualizar
            idx = np.random.default_rng(C.SEED).choice(len(faces), max_caras, replace=False)
            faces = faces[idx]
        tri = mesh.vertices[faces]
        col = Poly3DCollection(tri, alpha=alpha, facecolor=color, edgecolor="none")
        ax.add_collection3d(col)
    # límites a partir de las mallas presentes
    vts = np.vstack([m.vertices for m in (mesh_pred, mesh_gt) if m is not None])
    mn, mx = vts.min(0), vts.max(0)
    for set_lim, a, b in [(ax.set_xlim, mn[0], mx[0]), (ax.set_ylim, mn[1], mx[1]), (ax.set_zlim, mn[2], mx[2])]:
        set_lim(a, b)
    ax.set_box_aspect((mx - mn) if (mx - mn).min() > 0 else (1, 1, 1))
    ax.set_title(f"{cid}\nET: predicción (naranja) vs GT (verde)", fontsize=10)
    ax.axis("off")
    return R.guardar_figura(fig, figuras / f"render_{cid}.png")


# --------------------------------------------------------------------------- #
# Visor Three.js autocontenido (GLB embebidos en base64)
# --------------------------------------------------------------------------- #
def construir_visor(datos_casos, ruta_html: Path):
    """datos_casos: lista de dict {case_id, dice, pred_b64, gt_b64}. GLB embebidos."""
    data_js = json.dumps({
        d["case_id"]: {"dice": d["dice"], "pred": d["pred_b64"], "gt": d["gt_b64"]}
        for d in datos_casos
    })
    casos_js = json.dumps([d["case_id"] for d in datos_casos])
    html = _PLANTILLA_VISOR
    html = html.replace("__DATA__", data_js).replace("__CASES__", casos_js)
    html = html.replace("__COLOR_PRED__", COLOR_PRED).replace("__COLOR_GT__", COLOR_GT)
    ruta_html.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reporte HTML
# --------------------------------------------------------------------------- #
def construir_reporte(df, figs, ruta_html):
    validos = df[~df["omitido"]]
    mejor = validos.sort_values("dice", ascending=False).iloc[0] if len(validos) else None
    menor_dist = validos.sort_values("dist_media_mm").iloc[0] if len(validos) and validos["dist_media_mm"].notna().any() else None
    kpis = [
        R.kpi(str(df["case_id"].nunique()), "casos"),
        R.kpi(str(int((~df["omitido"]).sum()) * 2), "mallas (pred+gt)"),
        R.kpi(f"{mejor['dice']:.3f}" if mejor is not None else "—", "mejor Dice (caso)"),
        R.kpi(f"{menor_dist['dist_media_mm']:.2f} mm" if menor_dist is not None else "—",
              "menor dist. media sup."),
    ]
    secciones = []
    secciones.append(R.seccion(
        "Reconstrucción 3D de ET (predicción vs ground truth)",
        "<p>Mallas generadas con <b>marching cubes</b> sobre la máscara del método estrella "
        "<b>RegionGrowing</b> y sobre el <b>GT (seg==3)</b>, con spacing físico real (mm) y "
        "suavizado Taubin. Se exportan en <b>OBJ</b> (Slicer/Blender) y <b>GLB</b> (visor web). "
        "El visor interactivo está en <code>visor_3d.html</code> (Three.js local, sin internet).</p>"))
    secciones.append(R.seccion("Métricas 3D por caso", R.df_a_tabla_html(df.round(3))))
    bloques = [R.tarjeta_figura(R.png_a_base64(p), f"{cid}: malla ET pred (naranja) vs GT (verde).")
               for cid, p in figs.items()]
    secciones.append(R.seccion("Renders estáticos", *bloques))
    secciones.append(R.seccion(
        "Visor interactivo",
        "<p>Abre <code>output/reconstruccion3d/visor_3d.html</code> en un navegador "
        "(funciona offline). Selector de caso, pred (naranja sólido) + GT (verde translúcido) "
        "superpuestos, rotar/zoom/pan, toggles y el Dice del caso.</p>"))
    R.armar_reporte(
        "Reconstrucción 3D — ET · BraTS 2024 GLI",
        kpis, secciones,
        subtitulo="Módulo 6 · Mallas marching cubes + visor Three.js (pred vs GT)",
        ruta_salida=ruta_html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    base = C.OUTPUT_DIR / "reconstruccion3d"
    outputs = base / "outputs"; figuras = base / "figuras"
    outputs.mkdir(parents=True, exist_ok=True); figuras.mkdir(parents=True, exist_ok=True)
    seg_out = C.SALIDAS_MODULO["segmentacion"] / "outputs"

    info = json.loads((seg_out / "casos_segmentacion.json").read_text(encoding="utf-8"))
    casos = info["con_et"]
    detalle = pd.read_csv(seg_out / "metricas_segmentacion.csv")
    dice_rg = {r.case_id: r.dice for r in
               detalle[detalle.metodo == "RegionGrowing"].itertuples()}
    eda = pd.read_csv(C.SALIDAS_MODULO["eda"] / "outputs" / "estadisticas_por_caso.csv")
    vol_eda = {r.case_id: r.et_volumen_mm3 for r in eda.itertuples()}
    print(f"[3D] {len(casos)} casos con ET\n")

    filas, figs, datos_visor = [], {}, []
    for i, cid in enumerate(casos, 1):
        print(f"  [{i}/{len(casos)}] {cid}")
        img_pred = sitk.ReadImage(str(seg_out / f"{cid}-regiongrowing-ET.nii.gz"))
        spacing = img_pred.GetSpacing()
        mask_pred = io_zip.a_numpy(img_pred).astype(np.uint8)
        seg = io_zip.leer_seg_np(cid)
        mask_gt = (seg == C.LABEL_ET).astype(np.uint8)

        mesh_pred = malla_de_mascara(mask_pred, spacing)
        mesh_gt = malla_de_mascara(mask_gt, spacing)
        omitido = mesh_pred is None or mesh_gt is None
        if mesh_pred is None:
            print(f"        [AVISO] malla PRED omitida (vox={int(mask_pred.sum())} < {MIN_VOX})")
        if mesh_gt is None:
            print(f"        [AVISO] malla GT omitida (vox={int(mask_gt.sum())} < {MIN_VOX})")

        pred_b64 = gt_b64 = None
        if mesh_pred is not None:
            mesh_pred.export(str(outputs / f"{cid}-ET-pred.obj"))
            glb = mesh_pred.export(file_type="glb")
            (outputs / f"{cid}-ET-pred.glb").write_bytes(glb)
            pred_b64 = base64.b64encode(glb).decode("ascii")
        if mesh_gt is not None:
            mesh_gt.export(str(outputs / f"{cid}-ET-gt.obj"))
            glb = mesh_gt.export(file_type="glb")
            (outputs / f"{cid}-ET-gt.glb").write_bytes(glb)
            gt_b64 = base64.b64encode(glb).decode("ascii")

        dist_media = hausdorff = np.nan
        if mesh_pred is not None and mesh_gt is not None:
            dist_media, hausdorff = distancias_superficie(mesh_pred, mesh_gt)

        filas.append({
            "case_id": cid,
            "dice": round(float(dice_rg.get(cid, np.nan)), 4),
            "pred_vertices": 0 if mesh_pred is None else len(mesh_pred.vertices),
            "pred_caras": 0 if mesh_pred is None else len(mesh_pred.faces),
            "gt_vertices": 0 if mesh_gt is None else len(mesh_gt.vertices),
            "gt_caras": 0 if mesh_gt is None else len(mesh_gt.faces),
            "vol_pred_mm3": np.nan if mesh_pred is None else round(abs(float(mesh_pred.volume)), 1),
            "vol_gt_mm3": np.nan if mesh_gt is None else round(abs(float(mesh_gt.volume)), 1),
            "vol_gt_eda_mm3": round(float(vol_eda.get(cid, np.nan)), 1),
            "dist_media_mm": dist_media,
            "hausdorff_mm": hausdorff,
            "omitido": bool(omitido),
        })
        figs[cid] = render_estatico(cid, mesh_pred, mesh_gt, figuras)
        datos_visor.append({"case_id": cid, "dice": round(float(dice_rg.get(cid, np.nan)), 3),
                            "pred_b64": pred_b64, "gt_b64": gt_b64})
        nv_p = 0 if mesh_pred is None else len(mesh_pred.vertices)
        nv_g = 0 if mesh_gt is None else len(mesh_gt.vertices)
        print(f"        verts pred={nv_p} gt={nv_g}  vol pred/gt="
              f"{filas[-1]['vol_pred_mm3']}/{filas[-1]['vol_gt_mm3']} mm3  dist_media={dist_media}")
        for f in C.TMP_DIR.glob(f"{cid}-*.nii.gz"):
            f.unlink(missing_ok=True)

    df = pd.DataFrame(filas)
    df.to_csv(outputs / "metricas_3d.csv", index=False)

    print("\n[3D] generando visor y reporte...")
    construir_visor(datos_visor, base / "visor_3d.html")
    construir_reporte(df, figs, base / "reconstruccion3d_reporte.html")

    print("\n========== RESUMEN RECONSTRUCCIÓN 3D ==========")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[["case_id", "dice", "pred_vertices", "gt_vertices",
                  "vol_pred_mm3", "vol_gt_mm3", "vol_gt_eda_mm3", "dist_media_mm"]].to_string(index=False))
    print(f"\n  visor : {base / 'visor_3d.html'}")
    print(f"  salidas en: {base}")
    print("===============================================")


# --------------------------------------------------------------------------- #
# Plantilla del visor Three.js (GLB embebidos, librería local en lib/)
# --------------------------------------------------------------------------- #
_PLANTILLA_VISOR = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Visor 3D — ET (pred vs GT) · BraTS 2024 GLI</title>
<style>
  html,body{margin:0;height:100%;background:#0e1116;color:#e6edf3;font-family:system-ui,Segoe UI,sans-serif;overflow:hidden}
  #ui{position:absolute;top:12px;left:12px;z-index:10;background:rgba(20,26,34,.85);
      padding:12px 14px;border-radius:10px;border:1px solid #2a3340;max-width:300px}
  #ui h1{font-size:15px;margin:0 0 8px}
  select,button{font:inherit;background:#1b2330;color:#e6edf3;border:1px solid #34404f;
      border-radius:6px;padding:6px 8px;margin:3px 0}
  select{width:100%}
  label{display:block;margin:6px 0;font-size:13px}
  .sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:middle;margin-right:6px}
  #dice{font-size:13px;margin-top:8px;color:#9fb3c8}
  #msg{position:absolute;bottom:14px;left:12px;z-index:10;color:#ff9b9b;font-size:13px}
  #leyenda{position:absolute;top:12px;right:12px;z-index:10;background:rgba(20,26,34,.85);
      padding:10px 12px;border-radius:10px;border:1px solid #2a3340;font-size:12px}
</style>
</head>
<body>
<div id="ui">
  <h1>ET 3D — predicción vs GT</h1>
  <select id="caso"></select>
  <label><input type="checkbox" id="tPred" checked><span class="sw" style="background:__COLOR_PRED__"></span>Predicción (RegionGrowing)</label>
  <label><input type="checkbox" id="tGt" checked><span class="sw" style="background:__COLOR_GT__"></span>Ground truth</label>
  <button id="reset">Reset cámara</button>
  <div id="dice"></div>
</div>
<div id="leyenda">Arrastrar: rotar · Rueda: zoom · Click derecho: pan</div>
<div id="msg"></div>
<script src="lib/three.min.js"></script>
<script src="lib/OrbitControls.js"></script>
<script src="lib/GLTFLoader.js"></script>
<script>
const DATA = __DATA__;
const CASES = __CASES__;
const C_PRED = "__COLOR_PRED__", C_GT = "__COLOR_GT__";

let scene, camera, renderer, controls, grupo, loader;
let homePos = null, homeTarget = null;

function init(){
  scene = new THREE.Scene(); scene.background = new THREE.Color(0x0e1116);
  camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 5000);
  camera.position.set(120,90,120);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x202830, 1.0));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8); dir.position.set(1,1,1); scene.add(dir);
  loader = new THREE.GLTFLoader();
  grupo = new THREE.Group(); scene.add(grupo);

  const sel = document.getElementById('caso');
  CASES.forEach(c => { const o=document.createElement('option'); o.value=c; o.textContent=c; sel.appendChild(o); });
  sel.addEventListener('change', () => cargarCaso(sel.value));
  document.getElementById('tPred').addEventListener('change', e => toggle('pred', e.target.checked));
  document.getElementById('tGt').addEventListener('change', e => toggle('gt', e.target.checked));
  document.getElementById('reset').addEventListener('click', resetCam);
  window.addEventListener('resize', onResize);
  cargarCaso(CASES[0]);
  animate();
}

function b64ToArrayBuffer(b64){
  const bin = atob(b64); const len = bin.length; const bytes = new Uint8Array(len);
  for (let i=0;i<len;i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function limpiarGrupo(){ while(grupo.children.length) grupo.remove(grupo.children[0]); }

function addMalla(b64, tipo){
  return new Promise((resolve) => {
    if(!b64){ resolve(false); return; }
    try{
      loader.parse(b64ToArrayBuffer(b64), '', (gltf) => {
        const mat = tipo==='pred'
          ? new THREE.MeshStandardMaterial({color:C_PRED, metalness:0.1, roughness:0.6})
          : new THREE.MeshStandardMaterial({color:C_GT, transparent:true, opacity:0.4,
              depthWrite:false, metalness:0.1, roughness:0.7});
        gltf.scene.traverse(o => { if(o.isMesh){ o.material = mat; o.userData.tipo = tipo; } });
        gltf.scene.userData.tipo = tipo;
        grupo.add(gltf.scene);
        resolve(true);
      }, (err) => { resolve(false); });
    }catch(e){ resolve(false); }
  });
}

async function cargarCaso(cid){
  document.getElementById('msg').textContent = '';
  limpiarGrupo();
  const d = DATA[cid];
  document.getElementById('dice').textContent = (d && d.dice===d.dice) ? ('Dice (RegionGrowing): '+d.dice.toFixed(3)) : 'Dice: N/A';
  const okP = await addMalla(d ? d.pred : null, 'pred');
  const okG = await addMalla(d ? d.gt : null, 'gt');
  if(!okP && !okG){ document.getElementById('msg').textContent = 'No se pudo cargar ninguna malla de '+cid; return; }
  if(!okP) document.getElementById('msg').textContent = 'Malla de predicción no disponible.';
  if(!okG) document.getElementById('msg').textContent = 'Malla de GT no disponible.';
  toggle('pred', document.getElementById('tPred').checked);
  toggle('gt', document.getElementById('tGt').checked);
  encuadrar();
}

function toggle(tipo, visible){
  grupo.traverse(o => { if(o.userData && o.userData.tipo===tipo && o.isMesh) o.visible = visible; });
}

function encuadrar(){
  const box = new THREE.Box3().setFromObject(grupo);
  if(box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const s = box.getSize(new THREE.Vector3()).length() || 100;
  controls.target.copy(c);
  camera.position.set(c.x + s*0.9, c.y + s*0.7, c.z + s*0.9);
  camera.near = s/100; camera.far = s*100; camera.updateProjectionMatrix();
  homePos = camera.position.clone(); homeTarget = controls.target.clone();
  controls.update();
}

function resetCam(){ if(homePos){ camera.position.copy(homePos); controls.target.copy(homeTarget); controls.update(); } }
function onResize(){ camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }
function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
