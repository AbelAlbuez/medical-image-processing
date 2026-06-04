"""
reporte.py
==========
Helpers para producir las TRES salidas de cada módulo de forma consistente:

  1. .png sueltos en figuras/   -> guardar_figura(fig, ruta_png)
  2. .html autocontenido        -> armar_reporte(titulo, kpis, secciones)
  3. (.nii.gz los maneja io_zip.guardar_sitk en cada módulo)

Flujo de oro (NUNCA regenerar la imagen dos veces):
    ruta = guardar_figura(fig, figuras / "seg_otsu_<caso>.png")   # -> PNG suelto en disco
    data = png_a_base64(ruta)                                     # -> data-URI leído del PNG
    html_fig = tarjeta_figura(data, "pie de figura")              # -> se incrusta en el HTML

El HTML embebe en base64 los MISMOS PNG ya guardados; no reemplaza ni a los PNG ni a
los .nii.gz. Sin dependencias de exportación (nada de PDF/playwright/wkhtmltopdf).

Estilo idéntico entre módulos: hero degradado azul, fila de KPIs, secciones <h2> con
borde izquierdo azul, tablas estilo pandas. Paleta en constantes.PALETA.
"""
from __future__ import annotations

import base64
import html as _html
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")                 # backend sin pantalla (scripts, no notebooks)
import matplotlib.pyplot as plt       # noqa: E402  (import tras set backend, a propósito)
import pandas as pd

from . import constantes as C

P = C.PALETA


# --------------------------------------------------------------------------- #
# 1) Figuras: guardar como PNG suelto y leerlo como base64
# --------------------------------------------------------------------------- #
def guardar_figura(fig, ruta_png: Union[str, Path], dpi: int = 140,
                   cerrar: bool = True) -> Path:
    """
    Guarda una figura matplotlib como PNG suelto en disco (dpi>=130, bbox ajustado) y
    devuelve la ruta. Cierra la figura por defecto para no acumular memoria.
    """
    ruta_png = Path(ruta_png)
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_png, dpi=dpi, bbox_inches="tight")
    if cerrar:
        plt.close(fig)
    return ruta_png


def png_a_base64(ruta_png: Union[str, Path]) -> str:
    """Lee un PNG de disco y lo devuelve como data-URI base64 para embeber en HTML."""
    datos = Path(ruta_png).read_bytes()
    b64 = base64.b64encode(datos).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
# 2) Componentes HTML
# --------------------------------------------------------------------------- #
def df_a_tabla_html(df: pd.DataFrame, indice: bool = False,
                    na_rep: str = "N/A") -> str:
    """
    Convierte un DataFrame a una tabla HTML con estilo pandas (clase .tabla).
    NaN se muestra como 'N/A'. No usa pandas.Styler (sin dependencias extra).
    """
    return df.to_html(index=indice, na_rep=na_rep, border=0,
                      classes="tabla", justify="center", escape=True)


def tarjeta_figura(data_uri: str, pie: str = "") -> str:
    """Envuelve una figura (data-URI) en una tarjeta .fig con su pie."""
    pie_html = f'<div class="fig-pie">{_html.escape(pie)}</div>' if pie else ""
    return (f'<figure class="fig"><img src="{data_uri}" alt="{_html.escape(pie)}"/>'
            f'{pie_html}</figure>')


def seccion(titulo: str, *contenidos: str) -> str:
    """Una sección <h2> (borde izq azul) con su contenido HTML concatenado."""
    cuerpo = "\n".join(contenidos)
    return f'<section class="sec"><h2>{_html.escape(titulo)}</h2>\n{cuerpo}\n</section>'


def kpi(valor: str, etiqueta: str) -> Mapping[str, str]:
    """Construye un KPI {'valor','etiqueta'} para la fila de KPIs del hero."""
    return {"valor": valor, "etiqueta": etiqueta}


# --------------------------------------------------------------------------- #
# 3) Cascarón completo del reporte
# --------------------------------------------------------------------------- #
def armar_reporte(titulo: str, kpis: Sequence[Mapping[str, str]],
                  secciones: Iterable[str], subtitulo: str = "",
                  ruta_salida: Optional[Union[str, Path]] = None) -> str:
    """
    Ensambla el HTML autocontenido: hero (título+subtítulo) + fila de KPIs + secciones.
    Si `ruta_salida` se da, escribe el archivo. Devuelve siempre el HTML como string.

    Parameters
    ----------
    titulo : título del módulo (aparece en el hero y en <title>).
    kpis : lista de dicts {'valor','etiqueta'} (usar el helper kpi()). Se recomienda 4.
    secciones : iterable de bloques HTML (usar seccion()).
    subtitulo : línea bajo el título en el hero.
    ruta_salida : si se da, ruta del .html a escribir (NOMBRE PROPIO del módulo).
    """
    kpis_html = "\n".join(
        f'<div class="kpi"><div class="kpi-valor">{_html.escape(str(k["valor"]))}</div>'
        f'<div class="kpi-etq">{_html.escape(str(k["etiqueta"]))}</div></div>'
        for k in kpis
    )
    secciones_html = "\n".join(secciones)
    doc = _PLANTILLA.format(
        titulo=_html.escape(titulo),
        subtitulo=_html.escape(subtitulo),
        css=_css(),
        kpis=kpis_html,
        secciones=secciones_html,
    )
    if ruta_salida is not None:
        ruta_salida = Path(ruta_salida)
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        ruta_salida.write_text(doc, encoding="utf-8")
    return doc


def _css() -> str:
    """Hoja de estilo (idéntica entre módulos), parametrizada por la paleta."""
    return f"""
    :root {{
      --azul:{P['azul']}; --naranja:{P['naranja']}; --morado:{P['morado']};
      --verde:{P['verde']}; --rojo:{P['rojo']}; --tinta:{P['tinta']};
      --suave:{P['suave']}; --linea:{P['linea']}; --fondo:{P['fondo']};
      --cab:{P['tabla_cabecera']};
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--fondo); color:var(--tinta);
            font-family: system-ui,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
            line-height:1.55; }}
    .wrap {{ max-width:1180px; margin:0 auto; padding:0 22px 60px; }}
    .hero {{ background:linear-gradient(120deg,{P['hero_a']},{P['hero_b']});
             color:#fff; padding:38px 22px 30px; margin-bottom:26px; }}
    .hero .wrap {{ padding-bottom:0; }}
    .hero h1 {{ margin:0 0 6px; font-size:1.9rem; font-weight:700; }}
    .hero p {{ margin:0; opacity:.92; font-size:1.02rem; }}
    .kpis {{ display:flex; flex-wrap:wrap; gap:16px; margin:22px auto 0;
             max-width:1180px; padding:0 22px; }}
    .kpi {{ flex:1 1 0; min-width:170px; background:#fff; border-radius:12px;
            padding:16px 18px; box-shadow:0 1px 3px rgba(31,42,55,.08);
            border:1px solid var(--linea); }}
    .kpi-valor {{ font-size:1.5rem; font-weight:700; color:var(--azul); }}
    .kpi-etq {{ font-size:.82rem; color:var(--suave); text-transform:uppercase;
                letter-spacing:.03em; margin-top:2px; }}
    .sec {{ background:#fff; border:1px solid var(--linea); border-radius:12px;
            padding:18px 22px; margin:20px 0; box-shadow:0 1px 2px rgba(31,42,55,.05); }}
    .sec h2 {{ margin:0 0 14px; font-size:1.25rem; border-left:4px solid var(--azul);
               padding-left:10px; }}
    table.tabla {{ border-collapse:collapse; width:100%; font-size:.9rem; margin:6px 0 4px; }}
    table.tabla th {{ background:var(--cab); padding:8px 10px; text-align:center;
                      border-bottom:2px solid var(--linea); }}
    table.tabla td {{ padding:7px 10px; text-align:center; border-bottom:1px solid var(--linea); }}
    table.tabla tr:nth-child(even) td {{ background:#fafbfd; }}
    .fig {{ margin:14px 0; text-align:center; }}
    .fig img {{ max-width:100%; height:auto; border:1px solid var(--linea);
                border-radius:10px; background:#fff; }}
    .fig-pie {{ font-size:.85rem; color:var(--suave); margin-top:6px; }}
    """


_PLANTILLA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{titulo}</title>
<style>{css}</style>
</head>
<body>
<header class="hero">
  <div class="wrap"><h1>{titulo}</h1><p>{subtitulo}</p></div>
  <div class="kpis">{kpis}</div>
</header>
<main class="wrap">
{secciones}
</main>
</body>
</html>
"""
