# -*- coding: utf-8 -*-
"""
build_report.py
===============
Genera `report.html`: un panel interactivo y académico (en español) que resume
el proyecto de segmentación del Enhancing Tumor (ET) con métodos de contornos
deformables (spline / variational spline / b-spline / level set) sobre BraTS 2024.

Contenido:
  1. Intuición + matemática (MathJax) + animaciones (canvas) de cada método.
  2. Tablas de Dice/métricas por método, mapa de calor por caso y mejor método.
  3. Selector de tumor con la reconstrucción de superficie de Poisson
     (ground-truth vs segmentación) de los 5 mejores casos por Dice.

Las imágenes de Poisson se incrustan en base64 -> el HTML es autocontenido.
"""
from __future__ import annotations
import os, sys, glob, base64, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
CSV = os.path.join(ROOT, "output", "tablas", "metricas_ET.csv")
POIS = os.path.join(ROOT, "output", "figuras", "poisson")
OUT = os.path.join(ROOT, "report.html")

OURS = ["variational_spline", "bspline", "level_set", "spline"]
ETIQ = {
    "variational_spline": "Spline variacional (Chan–Vese)",
    "bspline": "B-spline (Chan–Vese + B-spline)",
    "level_set": "Level set (contorno activo geodésico)",
    "spline": "Spline (snake paramétrico)",
    "gmm_T1c": "GMM sobre T1c (clásico)",
    "otsu_T1c": "Otsu sobre T1c (clásico)",
    "sustraccion": "Sustracción T1c−T1n (clásico)",
    "gmm_2d": "GMM 2D (clásico)",
    "rango_doble": "Rango doble (clásico)",
    "fast_marching": "Fast Marching (clásico)",
}

# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
df = pd.read_csv(CSV)
THR = 0.75

agg = (df.groupby("metodo")["dice_ET"]
         .agg(media="mean", mediana="median", maximo="max", n="count")
         .round(3))
over = df[df.dice_ET > THR].groupby("metodo")["dice_ET"].count()
agg["sobre75"] = over.reindex(agg.index).fillna(0).astype(int)
agg = agg.sort_values("media", ascending=False)
mejor_metodo = agg.index[0]

# 5 mejores tumores (caso) según el MEJOR de nuestros 4 métodos, con imagen Poisson
sub = df[df.metodo.isin(OURS)].copy()
best_rows = sub.loc[sub.groupby("case_id")["dice_ET"].idxmax()]
best_rows = best_rows.sort_values("dice_ET", ascending=False)

poisson = []
for _, r in best_rows.iterrows():
    img = os.path.join(POIS, f"{r.case_id}_{r.metodo}_poisson.png")
    if os.path.exists(img):
        with open(img, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        poisson.append({
            "case": r.case_id, "metodo": r.metodo,
            "etiqueta": ETIQ.get(r.metodo, r.metodo),
            "dice": round(float(r.dice_ET), 3),
            "img": "data:image/png;base64," + b64,
        })
    if len(poisson) >= 5:
        break


# --------------------------------------------------------------------------- #
# Mallas 3D interactivas (Poisson) — GT y predicción, alineadas y empaquetadas
# como buffers binarios base64 (Float32 vértices, Uint16 índices) para Three.js
# --------------------------------------------------------------------------- #
import SimpleITK as _sitk
from brats_pipeline import viz_poisson as _VP


def _arr(path):
    return _sitk.GetArrayFromImage(_sitk.ReadImage(path))


def _mesh_va(mask, target=6000):
    """(vertices Nx3 en orden x,y,z, caras Mx3) de la malla de Poisson."""
    m = _VP._poisson_mesh(mask)
    if m is None or len(m.triangles) == 0:
        return None, None
    if len(m.triangles) > target:
        m = m.simplify_quadric_decimation(target)
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    if v.shape[0] == 0:
        return None, None
    return v[:, [2, 1, 0]], f          # (z,y,x) -> (x,y,z)


def _enc(v, f):
    fbits = 16 if v.shape[0] < 65535 else 32
    fb = f.astype("<u2" if fbits == 16 else "<u4").tobytes()
    return {
        "v": base64.b64encode(v.astype("<f4").tobytes()).decode("ascii"),
        "f": base64.b64encode(fb).decode("ascii"),
        "fbits": fbits, "nv": int(v.shape[0]), "nf": int(f.shape[0]),
    }


for p in poisson:
    case, met = p["case"], p["metodo"]
    try:
        pred = _arr(glob.glob(f"{ROOT}/output/segmentacion/{case}/{case}-et_{met}.nii.gz")[0]) > 0
        gt = np.round(_arr(glob.glob(f"{ROOT}/output/limpieza/{case}/*-seg.nii*")[0])) == 3
        vg, fg = _mesh_va(gt)
        vp, fp = _mesh_va(pred)
        if vg is None or vp is None:
            p["mesh"] = None
            continue
        allv = np.vstack([vg, vp])
        c = allv.mean(0)
        s = float(np.abs(allv - c).max()) or 1.0
        vg = (vg - c) / s * 100.0
        vp = (vp - c) / s * 100.0
        p["mesh"] = {"gt": _enc(vg, fg), "pred": _enc(vp, fp)}
        print(f"  malla 3D {case} {met}: GT {fg.shape[0]} tri, pred {fp.shape[0]} tri")
    except Exception as e:
        print(f"  [!] malla 3D {case} falló: {e}")
        p["mesh"] = None

# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def heat(d):
    """Color de fondo (blanco -> azul) y color de texto segun Dice in [0,1]."""
    d = max(0.0, min(1.0, float(d)))
    # blanco #ffffff -> azul #1f4e8c
    r = int(255 + (31 - 255) * d); g = int(255 + (78 - 255) * d); b = int(255 + (140 - 255) * d)
    txt = "#ffffff" if d > 0.55 else "#1e2a44"
    return f"background:rgb({r},{g},{b});color:{txt};"

# Tabla resumen por método
medalla = {0: "🥇", 1: "🥈", 2: "🥉"}
filas_metodos = []
for i, (m, row) in enumerate(agg.iterrows()):
    nuestro = m in OURS
    med = medalla.get(i, "")
    cls = "ours" if nuestro else "classic"
    barw = int(row.media * 100)
    barcol = "var(--purple-500)" if nuestro else "var(--sand-500)"
    filas_metodos.append(f"""
      <tr class="{cls}">
        <td class="rank">{med}</td>
        <td class="mname">{ETIQ.get(m, m)} <span class="mono">{m}</span>{' <span class="badge">NUEVO</span>' if nuestro else ''}</td>
        <td class="num"><div class="bar"><span style="width:{barw}%;background:{barcol}"></span></div><b>{row.media:.3f}</b></td>
        <td class="num">{row.mediana:.3f}</td>
        <td class="num">{row.maximo:.3f}</td>
        <td class="num">{row.sobre75}</td>
      </tr>""")
filas_metodos = "".join(filas_metodos)

# Mapa de calor por caso (nuestros 4 + 2 clásicos de referencia)
cols_hm = OURS + ["gmm_T1c", "otsu_T1c"]
piv = df.pivot_table(index="case_id", columns="metodo", values="dice_ET")
piv = piv.sort_values("variational_spline", ascending=False)
th_hm = "".join(f"<th>{ETIQ.get(c,c).split(' (')[0]}</th>" for c in cols_hm)
filas_hm = []
for case, row in piv.iterrows():
    vals = {c: row.get(c, float("nan")) for c in cols_hm}
    best_c = max(vals, key=lambda c: (vals[c] if vals[c] == vals[c] else -1))
    celdas = []
    for c in cols_hm:
        v = vals[c]
        if v != v:
            celdas.append('<td class="hm">—</td>')
        else:
            star = "★" if c == best_c else ""
            celdas.append(f'<td class="hm" style="{heat(v)}">{v:.2f}<sup>{star}</sup></td>')
    filas_hm.append(f'<tr><td class="case">{case.replace("BraTS-GLI-","")}</td>{"".join(celdas)}</tr>')
filas_hm = "".join(filas_hm)

datos_js = json.dumps(poisson, ensure_ascii=False)
n_over = int((df.dice_ET > THR).sum())
resumen = {
    "n_casos": int(df.case_id.nunique()),
    "mejor": ETIQ.get(mejor_metodo, mejor_metodo),
    "mejor_media": float(agg.loc[mejor_metodo, "media"]),
    "mejor_max": float(agg.loc[mejor_metodo, "maximo"]),
    "n_over": n_over,
}

# --------------------------------------------------------------------------- #
# Plantilla HTML  (tokens {{...}} se sustituyen; las llaves de CSS/JS quedan)
# --------------------------------------------------------------------------- #
TPL = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segmentación del Enhancing Tumor · Contornos deformables · BraTS 2024</title>
<script>
MathJax = { tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
:root{
  --blue-900:#13294b; --blue-700:#1f4e8c; --blue-500:#3a78c2; --blue-300:#7fb0e0;
  --blue-200:#cfe2f6; --blue-50:#eef5fc;
  --purple-700:#553a86; --purple-500:#7559ad; --purple-400:#9b7fc7; --purple-100:#ece4f6;
  --sand-600:#c08a2c; --sand-500:#d6a23e; --sand-300:#e9cd86; --sand-100:#f6eccf;
  --ink:#1e2a44; --muted:#5b6b86; --bg:#f5f9ff; --card:#ffffff; --line:#dde7f3;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;
  color:var(--ink);background:
    radial-gradient(1200px 600px at 80% -10%, var(--purple-100), transparent 60%),
    radial-gradient(1000px 500px at -10% 10%, var(--blue-50), transparent 55%),
    var(--bg);line-height:1.6}
a{color:var(--blue-700);text-decoration:none}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
/* Nav */
nav{position:sticky;top:0;z-index:50;backdrop-filter:blur(8px);
  background:rgba(255,255,255,.82);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:18px;align-items:center;height:58px;flex-wrap:wrap}
nav .brand{font-weight:700;color:var(--blue-900);letter-spacing:.2px}
nav .brand b{color:var(--purple-500)}
nav a.lnk{color:var(--muted);font-size:.92rem;padding:6px 10px;border-radius:8px}
nav a.lnk:hover{color:var(--blue-700);background:var(--blue-50)}
/* Hero */
header.hero{padding:54px 0 30px;
  background:linear-gradient(135deg,var(--blue-900),var(--blue-700) 45%,var(--purple-500));
  color:#fff;border-radius:0 0 26px 26px;box-shadow:0 18px 40px -28px var(--blue-900)}
header.hero h1{margin:0 0 8px;font-size:2.05rem;font-weight:800;letter-spacing:.3px}
header.hero p{margin:0;max-width:760px;color:#e7eefb;font-size:1.05rem}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin-top:24px}
.kpi{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);
  border-radius:14px;padding:14px 18px;min-width:150px}
.kpi .v{font-size:1.7rem;font-weight:800}
.kpi .l{font-size:.82rem;color:#dce7f8;text-transform:uppercase;letter-spacing:.5px}
/* Sections */
section{padding:40px 0 10px}
h2{font-size:1.5rem;color:var(--blue-900);margin:0 0 6px;
  display:flex;align-items:center;gap:10px}
h2 .dot{width:12px;height:12px;border-radius:50%;
  background:linear-gradient(var(--sand-500),var(--purple-500))}
.lead{color:var(--muted);max-width:850px;margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:22px;box-shadow:0 10px 30px -26px var(--blue-900);margin-bottom:20px}
/* Method grid */
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:820px){.mgrid{grid-template-columns:1fr}}
.method{border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#fff;
  display:flex;flex-direction:column}
.method .top{display:flex;align-items:center;gap:12px;padding:14px 16px;color:#fff}
.method.m1 .top{background:linear-gradient(120deg,var(--blue-700),var(--blue-500))}
.method.m2 .top{background:linear-gradient(120deg,var(--purple-700),var(--purple-400))}
.method.m3 .top{background:linear-gradient(120deg,var(--sand-600),var(--sand-500))}
.method.m4 .top{background:linear-gradient(120deg,#2f6e6b,#4aa39a)}
.method .top h3{margin:0;font-size:1.06rem}
.method .top .tag{margin-left:auto;font-size:.72rem;background:rgba(255,255,255,.22);
  padding:3px 9px;border-radius:20px;font-family:ui-monospace,Consolas,monospace}
.method .body{padding:16px}
.method canvas{width:100%;height:210px;display:block;background:
  radial-gradient(circle at 50% 45%, #fbfdff, #eef4fb);border-radius:12px;border:1px solid var(--line)}
.method .cap{font-size:.86rem;color:var(--muted);margin:10px 2px 0}
.math{background:var(--blue-50);border:1px solid var(--blue-200);border-radius:10px;
  padding:6px 12px;margin-top:12px;overflow-x:auto;font-size:.95rem}
.anim-ctl{display:flex;gap:8px;align-items:center;margin-top:10px}
.anim-ctl button{border:1px solid var(--line);background:#fff;border-radius:8px;
  padding:5px 12px;cursor:pointer;color:var(--blue-700);font-size:.85rem}
.anim-ctl button:hover{background:var(--blue-50)}
/* Tables */
table{width:100%;border-collapse:collapse;font-size:.92rem}
.tbl-methods th,.tbl-methods td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}
.tbl-methods th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.4px}
.tbl-methods td.num{text-align:right;white-space:nowrap}
.tbl-methods tr.ours{background:linear-gradient(90deg,var(--purple-100),transparent 60%)}
.tbl-methods .rank{font-size:1.15rem;text-align:center;width:42px}
.tbl-methods .mname .mono{font-family:ui-monospace,Consolas,monospace;color:var(--muted);font-size:.78rem;margin-left:6px}
.badge{background:var(--purple-500);color:#fff;font-size:.64rem;padding:2px 7px;border-radius:20px;
  vertical-align:middle;letter-spacing:.5px}
.bar{display:inline-block;width:90px;height:8px;border-radius:6px;background:var(--blue-50);
  vertical-align:middle;margin-right:8px;overflow:hidden}
.bar span{display:block;height:100%}
/* Heatmap */
.hmwrap{overflow-x:auto}
.tbl-hm{font-size:.82rem;min-width:620px}
.tbl-hm th,.tbl-hm td{padding:7px 8px;text-align:center;border:1px solid #eef3fa}
.tbl-hm th{color:var(--muted);font-weight:600;font-size:.74rem}
.tbl-hm td.case{font-family:ui-monospace,Consolas,monospace;color:var(--blue-900);
  background:var(--blue-50);font-weight:600;white-space:nowrap}
.tbl-hm td.hm{font-variant-numeric:tabular-nums}
.tbl-hm sup{color:var(--sand-300)}
.podium{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px}
.pod{flex:1;min-width:180px;border-radius:14px;padding:16px;color:#fff;
  background:linear-gradient(135deg,var(--purple-700),var(--purple-400))}
.pod.silver{background:linear-gradient(135deg,var(--blue-700),var(--blue-500))}
.pod.bronze{background:linear-gradient(135deg,var(--sand-600),var(--sand-500))}
.pod .p1{font-size:.8rem;opacity:.9}.pod .p2{font-size:1.05rem;font-weight:700;margin:2px 0}
.pod .p3{font-size:1.9rem;font-weight:800}
/* Poisson viewer */
.viewer{display:grid;grid-template-columns:300px 1fr;gap:22px}
@media(max-width:820px){.viewer{grid-template-columns:1fr}}
.controls label{display:block;font-size:.82rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.4px;margin-bottom:8px}
select{width:100%;padding:11px 12px;border-radius:10px;border:1px solid var(--line);
  background:#fff;color:var(--ink);font-size:.95rem}
.dicebig{margin-top:18px;border-radius:14px;padding:18px;text-align:center;
  background:linear-gradient(135deg,var(--blue-50),var(--purple-100));border:1px solid var(--line)}
.dicebig .num{font-size:2.6rem;font-weight:800;color:var(--blue-900)}
.dicebig .lab{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.dicebig .met{margin-top:6px;font-size:.9rem;color:var(--purple-700);font-weight:600}
.legend{display:flex;gap:18px;justify-content:center;margin-top:12px;font-size:.85rem}
.legend span{display:inline-flex;align-items:center;gap:6px}
.swatch{width:14px;height:14px;border-radius:4px;display:inline-block}
.stage{background:linear-gradient(160deg,#0e1b33,#16294a);border:1px solid var(--line);
  border-radius:16px;padding:0;overflow:hidden;
  display:flex;align-items:center;justify-content:center;min-height:420px;position:relative}
.stage img{max-width:100%;border-radius:10px;background:#fff}
#stage3d{width:100%;height:420px;cursor:grab}
#stage3d:active{cursor:grabbing}
#stage3d canvas{display:block}
.toggles{margin-top:16px;display:flex;flex-direction:column;gap:8px}
.toggles .ck{font-size:.9rem;color:var(--ink);display:flex;align-items:center;gap:8px;cursor:pointer}
.toggles .ck.rng{gap:10px}
.toggles .rng input[type=range]{flex:1;accent-color:var(--purple-500)}
.toggles input[type=checkbox]{accent-color:var(--blue-700);width:16px;height:16px}
.btnrow{display:flex;gap:8px;margin-top:4px;flex-wrap:wrap}
.btnrow button{border:1px solid var(--line);background:#fff;border-radius:9px;
  padding:7px 12px;cursor:pointer;color:var(--blue-700);font-size:.85rem;flex:1}
.btnrow button:hover{background:var(--blue-50)}
.loading3d{position:absolute;color:#cfe2f6;font-size:.95rem;letter-spacing:.3px}
footer{color:var(--muted);font-size:.85rem;padding:30px 0 50px;text-align:center}
.note{font-size:.85rem;color:var(--muted);background:var(--sand-100);
  border-left:4px solid var(--sand-500);padding:10px 14px;border-radius:8px;margin-top:14px}
.pill{display:inline-block;background:var(--blue-50);border:1px solid var(--blue-200);
  color:var(--blue-700);border-radius:20px;padding:2px 10px;font-size:.78rem;margin:2px}
</style>
</head>
<body>
<nav><div class="wrap">
  <div class="brand">BraTS&nbsp;2024 · <b>Contornos deformables</b></div>
  <a class="lnk" href="#intro">Intuición</a>
  <a class="lnk" href="#metodos">Métodos &amp; animaciones</a>
  <a class="lnk" href="#metricas">Métricas</a>
  <a class="lnk" href="#poisson">Superficies 3D</a>
</div></nav>

<header class="hero"><div class="wrap">
  <h1>Segmentación del <em>Enhancing Tumor</em> con modelos de contorno deformable</h1>
  <p>Spline · spline variacional · B-spline · level set — completamente automáticos sobre
     BraTS 2024 GLI. De una línea base clásica de Dice ≈ 0.3 a un refinamiento robusto
     inicializado por una semilla híbrida, con reconstrucción de superficie de Poisson.</p>
  <div class="kpis">
    <div class="kpi"><div class="v">{{N_CASOS}}</div><div class="l">Casos evaluados</div></div>
    <div class="kpi"><div class="v">{{MEJOR_MEDIA}}</div><div class="l">Dice medio · mejor método</div></div>
    <div class="kpi"><div class="v">{{MEJOR_MAX}}</div><div class="l">Dice máximo</div></div>
    <div class="kpi"><div class="v">{{N_OVER}}</div><div class="l">(caso,método) &gt; 0.75</div></div>
  </div>
</div></header>

<!-- ============ INTRO ============ -->
<section id="intro"><div class="wrap">
  <h2><span class="dot"></span>Intuición general</h2>
  <p class="lead">El <b>Enhancing Tumor</b> (ET, etiqueta 3) es el borde tumoral que capta
    gadolinio: brillante en T1c y con realce positivo en el mapa de diferencia
    \(D = \text{T1c}-\text{T1n}\). Es la subregión más difícil de BraTS. En lugar de
    umbralizar, tratamos la segmentación como la <b>evolución de un contorno</b> que parte
    de una semilla automática y se deforma hacia el borde real del realce minimizando una
    energía. Cuatro modelos clásicos materializan esa idea con regularizaciones distintas.</p>
  <div class="card">
    <b>Semilla híbrida automática.</b> Ninguna señal aislada es robusta para el ET, así que
    cada caso parte de la semilla más fiable: un <b>GMM de 3 componentes sobre T1c</b>
    (localiza la masa realzante aun con mapa de diferencia débil) y, cuando éste degenera,
    el <b>blob dominante del mapa \(D\)</b>. Los cuatro modelos refinan esa semilla; una
    salvaguarda revierte la evolución si colapsa, se fuga o se va a otra región — de modo
    que un modelo nunca empeora la base.
    <div class="note">Convención de color en todo el panel: <span class="pill">azules = level set / datos</span>
      <span class="pill" style="background:var(--purple-100);border-color:var(--purple-400);color:var(--purple-700)">púrpura = métodos nuevos</span>
      <span class="pill" style="background:var(--sand-100);border-color:var(--sand-500);color:var(--sand-600)">arena = referencia clásica</span></div>
  </div>
</div></section>

<!-- ============ MÉTODOS ============ -->
<section id="metodos"><div class="wrap">
  <h2><span class="dot"></span>Los cuatro métodos: intuición, matemática y animación</h2>
  <p class="lead">Cada tarjeta muestra una animación esquemática de cómo el contorno (curva)
    evoluciona desde la inicialización hasta ajustarse al tumor, junto con la energía o
    ecuación que gobierna ese movimiento. Pasa el cursor o usa ▮▮/▶ para pausar.</p>
  <div class="mgrid">

    <div class="method m1">
      <div class="top"><h3>Level set — contorno activo geodésico</h3><span class="tag">level_set</span></div>
      <div class="body">
        <canvas id="cv_level" width="520" height="300"></canvas>
        <div class="anim-ctl"><button data-cv="level">▮▮ / ▶</button>
          <span class="cap">Un frente implícito \(\phi\) avanza rápido en zonas planas y se
          frena donde el gradiente es alto (borde del realce).</span></div>
        <div class="math">\[ \frac{\partial \phi}{\partial t}= g(|\nabla I|)\,\big(\alpha+\beta\,\kappa\big)\,|\nabla\phi| + \nabla g\cdot\nabla\phi,\quad g=\frac{1}{1+|\nabla (G_\sigma * I)|}\]</div>
      </div>
    </div>

    <div class="method m2">
      <div class="top"><h3>Spline variacional — Chan–Vese</h3><span class="tag">variational_spline</span></div>
      <div class="body">
        <canvas id="cv_chan" width="520" height="300"></canvas>
        <div class="anim-ctl"><button data-cv="chan">▮▮ / ▶</button>
          <span class="cap">Sin bordes: el contorno separa la imagen en interior (media \(c_1\))
          y exterior (media \(c_2\)); la curvatura aporta suavidad tipo spline.</span></div>
        <div class="math">\[ E(c_1,c_2,C)=\mu\,\mathrm{Long}(C)+\lambda_1\!\!\int_{in}\!\!|I-c_1|^2 + \lambda_2\!\!\int_{out}\!\!|I-c_2|^2 \]</div>
      </div>
    </div>

    <div class="method m4">
      <div class="top"><h3>Spline — snake paramétrico (Kass)</h3><span class="tag">spline</span></div>
      <div class="body">
        <canvas id="cv_snake" width="520" height="300"></canvas>
        <div class="anim-ctl"><button data-cv="snake">▮▮ / ▶</button>
          <span class="cap">Una curva spline con nodos se contrae hacia el borde: la energía
          interna (tensión \(\alpha\), rigidez \(\beta\)) la mantiene suave.</span></div>
        <div class="math">\[ E=\!\int_0^1\! \tfrac12\big(\alpha|C'(s)|^2+\beta|C''(s)|^2\big) + E_{\text{ext}}\!\big(C(s)\big)\,ds \]</div>
      </div>
    </div>

    <div class="method m3">
      <div class="top"><h3>B-spline — frontera suavizada</h3><span class="tag">bspline</span></div>
      <div class="body">
        <canvas id="cv_bspline" width="520" height="300"></canvas>
        <div class="anim-ctl"><button data-cv="bspline">▮▮ / ▶</button>
          <span class="cap">La frontera dentada del level set se reescribe como un B-spline
          cúbico periódico (continuidad \(C^2\)): superficie lisa, sin escalones.</span></div>
        <div class="math">\[ C(u)=\sum_{i} N_{i,3}(u)\,P_i,\qquad u\in[0,1)\ \text{(periódico)} \]</div>
      </div>
    </div>

  </div>
</div></section>

<!-- ============ MÉTRICAS ============ -->
<section id="metricas"><div class="wrap">
  <h2><span class="dot"></span>Resultados: Dice y métricas por método</h2>
  <p class="lead">Evaluación sobre {{N_CASOS}} casos ricos en ET, totalmente automática.
    El <b>spline variacional (Chan–Vese)</b> es el mejor método y supera a todas las líneas
    base clásicas. Métrica: \( \mathrm{Dice}= \dfrac{2|A\cap B|}{|A|+|B|}\).</p>

  <div class="podium">
    <div class="pod"><div class="p1">🥇 Mejor método</div><div class="p2">Spline variacional</div><div class="p3">{{MEJOR_MEDIA}}</div><div class="p1">Dice medio · máx {{MEJOR_MAX}}</div></div>
    <div class="pod silver"><div class="p1">Salto vs. línea base</div><div class="p2">0.30 → {{MEJOR_MEDIA}}</div><div class="p3">+{{DELTA}}</div><div class="p1">mejora del Dice medio</div></div>
    <div class="pod bronze"><div class="p1">Superficies de Poisson</div><div class="p2">casos &gt; 0.75</div><div class="p3">{{N_OVER}}</div><div class="p1">(caso, método) reconstruidos</div></div>
  </div>

  <div class="card" style="margin-top:20px">
    <table class="tbl-methods">
      <thead><tr><th></th><th>Método</th><th class="num">Dice medio</th><th class="num">Mediana</th><th class="num">Máx</th><th class="num">Casos&gt;0.75</th></tr></thead>
      <tbody>{{FILAS_METODOS}}</tbody>
    </table>
  </div>

  <h3 style="color:var(--blue-900);margin:6px 0 10px">Mapa de calor — Dice por caso</h3>
  <p class="lead" style="margin-bottom:12px">Cada celda es el Dice de un caso (fila) con un
    método (columna); ★ marca el mejor método de ese caso. Azul más intenso = mejor.</p>
  <div class="card hmwrap">
    <table class="tbl-hm">
      <thead><tr><th>Caso</th>{{TH_HM}}</tr></thead>
      <tbody>{{FILAS_HM}}</tbody>
    </table>
  </div>
</div></section>

<!-- ============ POISSON ============ -->
<section id="poisson"><div class="wrap">
  <h2><span class="dot"></span>Reconstrucción de superficie de Poisson — 5 mejores tumores</h2>
  <p class="lead">Para los casos con Dice &gt; 0.75 reconstruimos la superficie 3D del tumor
    (Poisson screened, open3d) y la comparamos con el ground-truth. Elige un tumor en el
    menú para ver <b>verde = ground-truth</b> frente a <b>rojo = segmentación</b>.</p>
  <div class="card viewer">
    <div class="controls">
      <label for="sel">Selecciona el tumor</label>
      <select id="sel"></select>
      <div class="dicebig">
        <div class="num" id="dval">—</div>
        <div class="lab">Dice ET</div>
        <div class="met" id="mval">—</div>
      </div>
      <div class="legend">
        <span><i class="swatch" style="background:#33bf4d"></i>Ground-truth</span>
        <span><i class="swatch" style="background:#d93333"></i>Segmentación</span>
      </div>
      <div class="toggles">
        <label class="ck"><input type="checkbox" id="ckgt" checked> Mostrar GT</label>
        <label class="ck"><input type="checkbox" id="ckpr" checked> Mostrar segmentación</label>
        <label class="ck rng">Opacidad <input type="range" id="opac" min="20" max="100" value="80"></label>
        <div class="btnrow">
          <button id="reset3d">↺ Reiniciar vista</button>
          <button id="mode2d">🖼 Ver 2D</button>
        </div>
      </div>
      <div class="note" id="hint3d">🖱️ Arrastra para rotar · rueda para acercar/alejar ·
        clic derecho para desplazar. Mallas <code>.ply</code> en
        <code>output/figuras/poisson/</code>.</div>
    </div>
    <div class="stage">
      <div id="stage3d"></div>
      <img id="pimg" alt="Superficie de Poisson" style="display:none">
    </div>
  </div>
</div></section>

<footer><div class="wrap">
  Panel generado automáticamente a partir de <code>output/tablas/metricas_ET.csv</code> ·
  BraTS 2024 GLI · Segmentación clásica del Enhancing Tumor con contornos deformables.
</div></footer>

<script>
/* ====================== Visor de Poisson ====================== */
const POIS = {{DATOS_JS}};
const sel = document.getElementById('sel');
const pimg = document.getElementById('pimg');
const dval = document.getElementById('dval');
const mval = document.getElementById('mval');
const stage3d = document.getElementById('stage3d');
POIS.forEach((p,i)=>{
  const o=document.createElement('option');
  o.value=i; o.textContent=`${p.case.replace('BraTS-GLI-','Tumor ')} — Dice ${p.dice.toFixed(3)}`;
  sel.appendChild(o);
});

/* --- decodificación base64 -> typed arrays --- */
function b64bytes(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
function f32(b64){return new Float32Array(b64bytes(b64).buffer);}
function faceIdx(o){const u=b64bytes(o.f);return o.fbits===16?new Uint16Array(u.buffer):new Uint32Array(u.buffer);}

/* --- escena Three.js --- */
let renderer,scene,camera,controls,gtMesh,prMesh,has3d=false,mode='3d';
const CAM0=[170,115,170];
function init3d(){
  try{
    if(!window.THREE||!THREE.OrbitControls) return false;
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
    scene=new THREE.Scene();
    camera=new THREE.PerspectiveCamera(45,1,0.1,4000);
    camera.position.set(...CAM0);
    controls=new THREE.OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true;controls.dampingFactor=0.08;controls.rotateSpeed=0.9;
    scene.add(new THREE.AmbientLight(0xffffff,0.7));
    const d1=new THREE.DirectionalLight(0xffffff,0.85);d1.position.set(1,1.2,1);scene.add(d1);
    const d2=new THREE.DirectionalLight(0xbcd0ff,0.4);d2.position.set(-1,-0.6,-1);scene.add(d2);
    stage3d.appendChild(renderer.domElement);
    resize3d(); animate3d(); has3d=true; return true;
  }catch(e){return false;}
}
function resize3d(){if(!renderer)return;const w=stage3d.clientWidth||600,h=stage3d.clientHeight||420;
  renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
window.addEventListener('resize',resize3d);
function animate3d(){requestAnimationFrame(animate3d);if(controls)controls.update();
  if(renderer&&scene&&camera)renderer.render(scene,camera);}
function makeMesh(o,color){
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(f32(o.v),3));
  g.setIndex(new THREE.BufferAttribute(faceIdx(o),1));
  g.computeVertexNormals();
  return new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:color,roughness:0.55,
    metalness:0.05,transparent:true,opacity:0.8,side:THREE.DoubleSide}));
}
function clearMesh(m){if(m){scene.remove(m);m.geometry.dispose();m.material.dispose();}}
function load3d(p){
  clearMesh(gtMesh);clearMesh(prMesh);gtMesh=prMesh=null;
  if(!p.mesh)return false;
  gtMesh=makeMesh(p.mesh.gt,0x33bf4d);prMesh=makeMesh(p.mesh.pred,0xd93333);
  scene.add(gtMesh);scene.add(prMesh);applyVis();return true;
}
function applyVis(){const op=document.getElementById('opac').value/100;
  if(gtMesh){gtMesh.visible=document.getElementById('ckgt').checked;gtMesh.material.opacity=op;}
  if(prMesh){prMesh.visible=document.getElementById('ckpr').checked;prMesh.material.opacity=op;}}
function resetView(){if(controls){controls.target.set(0,0,0);camera.position.set(...CAM0);controls.update();}}
function setMode(m){mode=m;const is3=(m==='3d');
  stage3d.style.display=is3?'block':'none';pimg.style.display=is3?'none':'block';
  document.getElementById('mode2d').textContent=is3?'🖼 Ver 2D':'🧊 Ver 3D';
  document.getElementById('hint3d').style.display=is3?'block':'none';
  if(is3)resize3d();}
function show(i){const p=POIS[i];pimg.src=p.img;dval.textContent=p.dice.toFixed(3);mval.textContent=p.etiqueta;
  if(has3d){const ok=load3d(p);
    if(ok){resetView();setMode('3d');document.getElementById('mode2d').style.display='';}
    else{setMode('2d');document.getElementById('mode2d').style.display='none';}
  }}
['ckgt','ckpr'].forEach(id=>document.getElementById(id).addEventListener('change',applyVis));
document.getElementById('opac').addEventListener('input',applyVis);
document.getElementById('reset3d').addEventListener('click',resetView);
document.getElementById('mode2d').addEventListener('click',()=>setMode(mode==='3d'?'2d':'3d'));

if(!init3d()){            /* sin WebGL: degradar a 2D */
  has3d=false;stage3d.style.display='none';pimg.style.display='block';
  document.querySelector('.toggles').style.display='none';
  document.getElementById('hint3d').textContent='Vista 2D estática (no se pudo iniciar el visor 3D / WebGL).';
}
sel.addEventListener('change',e=>show(+e.target.value));
if(POIS.length) show(0);

/* ====================== Animaciones (canvas) ====================== */
// Blob tumoral compartido: radio por ángulo (irregular, con lóbulos)
function targetR(a){return 1 + 0.22*Math.sin(3*a+0.6) + 0.12*Math.sin(5*a+1.7) + 0.07*Math.cos(2*a);}
function smoothClosed(ctx, pts){
  // Catmull-Rom cerrada -> curva suave
  const n=pts.length; ctx.beginPath();
  for(let i=0;i<n;i++){
    const p0=pts[(i-1+n)%n],p1=pts[i],p2=pts[(i+1)%n],p3=pts[(i+2)%n];
    if(i===0)ctx.moveTo(p1.x,p1.y);
    for(let t=0;t<1;t+=0.2){
      const t2=t*t,t3=t2*t;
      const x=0.5*((2*p1.x)+(-p0.x+p2.x)*t+(2*p0.x-5*p1.x+4*p2.x-p3.x)*t2+(-p0.x+3*p1.x-3*p2.x+p3.x)*t3);
      const y=0.5*((2*p1.y)+(-p0.y+p2.y)*t+(2*p0.y-5*p1.y+4*p2.y-p3.y)*t2+(-p0.y+3*p1.y-3*p2.y+p3.y)*t3);
      ctx.lineTo(x,y);
    }
  }
  ctx.closePath();
}
function drawTumor(ctx,cx,cy,R){
  // relleno arena con núcleo brillante = realce
  const pts=[];for(let k=0;k<60;k++){const a=k/60*2*Math.PI,r=R*targetR(a);pts.push({x:cx+r*Math.cos(a),y:cy+r*Math.sin(a)});}
  smoothClosed(ctx,pts);
  const g=ctx.createRadialGradient(cx,cy,R*0.1,cx,cy,R*1.25);
  g.addColorStop(0,'#f6eccf');g.addColorStop(0.6,'#ecd49a');g.addColorStop(1,'#e3c77f');
  ctx.fillStyle=g;ctx.fill();
  ctx.lineWidth=1;ctx.strokeStyle='rgba(192,138,44,.55)';ctx.stroke();
}
const anims={};
function reg(id,fn){const cv=document.getElementById(id);const ctx=cv.getContext('2d');
  anims[id]={cv,ctx,fn,t:0,run:true};}
reg('cv_level',drawLevel); reg('cv_chan',drawChan); reg('cv_snake',drawSnake); reg('cv_bspline',drawBspline);

function frameCommon(a){const {ctx,cv}=a;ctx.clearRect(0,0,cv.width,cv.height);return {cx:cv.width/2,cy:cv.height/2,R:Math.min(cv.width,cv.height)*0.30};}

// ---- LEVEL SET: frente que crece y se detiene en el borde
function drawLevel(a){const {ctx}=a;const {cx,cy,R}=frameCommon(a);
  drawTumor(ctx,cx,cy,R);
  const t=(a.t%240)/240; const grow=Math.min(1, t*1.6); // 0..1
  const pts=[];for(let k=0;k<60;k++){const ang=k/60*2*Math.PI;
    const rt=R*targetR(ang); const r0=R*0.18;
    // el frente interpola de un círculo pequeño al borde, con leve curvatura
    let r=r0+(rt-r0)*easeOut(grow); r+=Math.sin(ang*6+a.t*0.05)*1.2*(1-grow);
    pts.push({x:cx+r*Math.cos(ang),y:cy+r*Math.sin(ang)});}
  ctx.lineWidth=3;ctx.strokeStyle='#1f4e8c';smoothClosed(ctx,pts);ctx.stroke();
  // flechas de propagación
  if(grow<0.98){ctx.strokeStyle='rgba(58,120,194,.5)';ctx.lineWidth=1.4;
    for(let k=0;k<12;k++){const ang=k/12*2*Math.PI;const r=R*0.18+(R*targetR(ang)-R*0.18)*easeOut(grow);
      ctx.beginPath();ctx.moveTo(cx+r*Math.cos(ang),cy+r*Math.sin(ang));
      ctx.lineTo(cx+(r+10)*Math.cos(ang),cy+(r+10)*Math.sin(ang));ctx.stroke();}}
  label(ctx,'frente φ=0  ·  velocidad g(|∇I|)', '#1f4e8c');
}
// ---- CHAN-VESE: contorno que separa interior (c1) / exterior (c2)
function drawChan(a){const {ctx}=a;const {cx,cy,R}=frameCommon(a);
  drawTumor(ctx,cx,cy,R);
  const t=(a.t%260)/260; const conv=easeInOut(Math.min(1,t*1.5));
  const pts=[];for(let k=0;k<60;k++){const ang=k/60*2*Math.PI;
    const rt=R*targetR(ang); const r0=R*(1.35+0.15*Math.sin(ang*2)); // empieza como óvalo externo
    const r=r0+(rt-r0)*conv;
    pts.push({x:cx+r*Math.cos(ang),y:cy+r*Math.sin(ang)});}
  // sombrea interior
  smoothClosed(ctx,pts);ctx.fillStyle='rgba(117,89,173,.16)';ctx.fill();
  ctx.lineWidth=3;ctx.strokeStyle='#7559ad';smoothClosed(ctx,pts);ctx.stroke();
  chip(ctx,cx-6,cy,'c₁','#7559ad'); chip(ctx,cx+R*1.45,cy-R*0.9,'c₂','#9aa7bd');
  label(ctx,'minimiza  μ·Long(C)+λ‖I−c₁‖²+λ‖I−c₂‖²','#553a86');
}
// ---- SNAKE: nodos spline que se contraen al borde
function drawSnake(a){const {ctx}=a;const {cx,cy,R}=frameCommon(a);
  drawTumor(ctx,cx,cy,R);
  const t=(a.t%240)/240; const conv=easeOut(Math.min(1,t*1.5));
  const N=16,pts=[];
  for(let k=0;k<N;k++){const ang=k/N*2*Math.PI;
    const rt=R*targetR(ang); const r0=R*1.42;
    const r=r0+(rt-r0)*conv; pts.push({x:cx+r*Math.cos(ang),y:cy+r*Math.sin(ang)});}
  ctx.lineWidth=2.5;ctx.strokeStyle='#553a86';smoothClosed(ctx,pts);ctx.stroke();
  // nodos
  for(const p of pts){ctx.beginPath();ctx.arc(p.x,p.y,3.6,0,2*Math.PI);
    ctx.fillStyle='#9b7fc7';ctx.fill();ctx.lineWidth=1.4;ctx.strokeStyle='#553a86';ctx.stroke();}
  label(ctx,'energía interna α|C′|²+β|C″|²  →  curva suave','#553a86');
}
// ---- B-SPLINE: dentado -> suave
function drawBspline(a){const {ctx}=a;const {cx,cy,R}=frameCommon(a);
  drawTumor(ctx,cx,cy,R);
  const t=(a.t%260)/260; const s=easeInOut(Math.min(1,t*1.4)); // 0 dentado -> 1 suave
  const N=42,raw=[];
  for(let k=0;k<N;k++){const ang=k/N*2*Math.PI;
    const rt=R*targetR(ang);
    const jag=Math.sin(ang*18)*6*(1-s)+ (Math.random()<0? 0:0); // dentado decreciente
    raw.push({x:cx+(rt+jag)*Math.cos(ang),y:cy+(rt+jag)*Math.sin(ang)});}
  // contorno dentado (gris) que se desvanece
  if(s<0.95){ctx.lineWidth=1.6;ctx.strokeStyle=`rgba(120,135,160,${0.6*(1-s)})`;
    ctx.beginPath();raw.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.stroke();}
  // B-spline suave (puntos de control + curva)
  const ctrl=[];for(let k=0;k<14;k++){const ang=k/14*2*Math.PI;const rt=R*targetR(ang);
    ctrl.push({x:cx+rt*Math.cos(ang),y:cy+rt*Math.sin(ang)});}
  ctx.setLineDash([3,4]);ctx.lineWidth=1;ctx.strokeStyle='rgba(214,162,62,.7)';
  ctx.beginPath();ctrl.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.stroke();ctx.setLineDash([]);
  for(const p of ctrl){ctx.beginPath();ctx.arc(p.x,p.y,3,0,2*Math.PI);ctx.fillStyle='#d6a23e';ctx.fill();}
  ctx.lineWidth=3;ctx.strokeStyle='#c08a2c';smoothClosed(ctx,ctrl);ctx.stroke();
  label(ctx,'C(u)=Σ Nᵢ,₃(u) Pᵢ   ·   continuidad C²','#c08a2c');
}
function easeOut(x){return 1-Math.pow(1-x,3);}
function easeInOut(x){return x<.5?4*x*x*x:1-Math.pow(-2*x+2,3)/2;}
function label(ctx,txt,col){ctx.font='12px Segoe UI, sans-serif';ctx.fillStyle=col;
  ctx.textAlign='center';ctx.fillText(txt,ctx.canvas.width/2,ctx.canvas.height-12);}
function chip(ctx,x,y,txt,col){ctx.font='bold 13px Segoe UI';ctx.fillStyle=col;ctx.textAlign='center';ctx.fillText(txt,x,y);}

// bucle
let last=0;
function loop(ts){const dt=ts-last;last=ts;
  for(const id in anims){const a=anims[id];if(a.run){a.t+=dt*0.06;a.fn(a);}}
  requestAnimationFrame(loop);}
requestAnimationFrame(loop);
// pausa/play
document.querySelectorAll('.anim-ctl button').forEach(b=>{
  b.addEventListener('click',()=>{const id='cv_'+b.dataset.cv;anims[id].run=!anims[id].run;});});
// pausa al salir del viewport (rendimiento)
const io=new IntersectionObserver(es=>es.forEach(e=>{const c=e.target.id;
  if(anims[c])anims[c].run=e.isIntersecting;}),{threshold:0.1});
['cv_level','cv_chan','cv_snake','cv_bspline'].forEach(id=>io.observe(document.getElementById(id)));
</script>
</body>
</html>"""

# Sustituciones
delta = round(resumen["mejor_media"] - 0.30, 3)
html = (TPL
        .replace("{{N_CASOS}}", str(resumen["n_casos"]))
        .replace("{{MEJOR_MEDIA}}", f'{resumen["mejor_media"]:.3f}')
        .replace("{{MEJOR_MAX}}", f'{resumen["mejor_max"]:.3f}')
        .replace("{{N_OVER}}", str(resumen["n_over"]))
        .replace("{{DELTA}}", f'{delta:.3f}')
        .replace("{{FILAS_METODOS}}", filas_metodos)
        .replace("{{TH_HM}}", th_hm)
        .replace("{{FILAS_HM}}", filas_hm)
        .replace("{{DATOS_JS}}", datos_js))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(html)
print(f"OK -> {OUT}  ({len(html)//1024} KB, {len(poisson)} superficies)")
