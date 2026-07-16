from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NODES_PATH = BASE_DIR / "nodes_final.geojson"
DEFAULT_LINKS_PATH = BASE_DIR / "links_final.geojson"
DEFAULT_HTML_PATH = BASE_DIR / "road_map_visualization.html"
DEFAULT_SVG_PATH = BASE_DIR / "road_map_visualization.svg"
DEFAULT_STATION_SETTINGS_PATH = BASE_DIR.parents[1] / "exps" / "data" / "setting_72stations_roadmap_pv_ess.yaml"
WEB_MERCATOR_RADIUS_M = 6378137.0

NODE_STYLES = {
    "toll": {"label": "Toll", "color": "#2364aa", "radius": 3.2},
    "interchange": {"label": "Interchange", "color": "#b8581a", "radius": 3.8},
    "service": {"label": "Service", "color": "#228b5a", "radius": 5.0},
    "unknown": {"label": "Unknown", "color": "#606b78", "radius": 3.0},
}


def web_mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = (float(x) / WEB_MERCATOR_RADIUS_M) * 180.0 / math.pi
    lat = (
        2.0 * math.atan(math.exp(float(y) / WEB_MERCATOR_RADIUS_M)) - math.pi / 2.0
    ) * 180.0 / math.pi
    return _zero_snap(lon), _zero_snap(lat)


def load_renewable_station_ids(settings_path: Path) -> set[int]:
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    station_ids = [int(item) for item in settings["station_ids"]]
    pv = [int(item) for item in settings["renewable"]["pv_indicator"]]
    ess = [int(item) for item in settings["ess"]["ess_indicator"]]
    if len(station_ids) != len(pv) or len(station_ids) != len(ess):
        raise ValueError("station_ids, pv_indicator, and ess_indicator must have equal lengths.")
    return {station_id for station_id, has_pv, has_ess in zip(station_ids, pv, ess) if has_pv or has_ess}


def load_network(
    nodes_path: Path,
    links_path: Path,
    renewable_station_ids: set[int] | None = None,
) -> dict[str, Any]:
    renewable_station_ids = renewable_station_ids or set()
    nodes_data = _read_feature_collection(nodes_path)
    links_data = _read_feature_collection(links_path)
    node_features = nodes_data.get("features", [])
    link_features = links_data.get("features", [])

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []

    for feature in node_features:
        properties = feature.get("properties", {})
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if not _is_xy(coordinates):
            continue
        x, y = float(coordinates[0]), float(coordinates[1])
        lon, lat = web_mercator_to_lonlat(x, y)
        node_type = str(properties.get("type") or "unknown")
        node_id = properties.get("id")
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "renewable": int(node_id) in renewable_station_ids if node_id is not None else False,
                "x": x,
                "y": y,
                "lon": lon,
                "lat": lat,
            }
        )
        xs.append(x)
        ys.append(y)

    for feature in link_features:
        properties = feature.get("properties", {})
        points = [(float(x), float(y)) for x, y in _iter_xy(feature.get("geometry", {}).get("coordinates", []))]
        if not points:
            continue
        raw_length = properties.get("length")
        source_length = raw_length if isinstance(raw_length, (int, float)) and not isinstance(raw_length, bool) else None
        length = source_length if source_length is not None else _polyline_length(points)
        length_source = "source" if source_length is not None else "computed_from_geometry"
        links.append(
            {
                "from_id": properties.get("from_id"),
                "to_id": properties.get("to_id"),
                "length": length,
                "length_source": length_source,
                "length_text": _format_length(length),
                "points": points,
            }
        )
        for x, y in points:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        raise ValueError("No drawable coordinates found in the road-map GeoJSON files.")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_lon, min_lat = web_mercator_to_lonlat(min_x, min_y)
    max_lon, max_lat = web_mercator_to_lonlat(max_x, max_y)
    node_ids = {node["id"] for node in nodes}
    known_lengths = [link["length"] for link in links if link["length"] is not None]
    computed_length_count = sum(1 for link in links if link["length_source"] == "computed_from_geometry")
    node_type_counts = dict(Counter(node["type"] for node in nodes))

    return {
        "nodes": nodes,
        "links": links,
        "stats": {
            "node_count": len(nodes),
            "link_count": len(links),
            "node_type_counts": node_type_counts,
            "renewable_station_count": sum(1 for node in nodes if node["renewable"]),
            "mercator_bbox": [min_x, min_y, max_x, max_y],
            "lonlat_bbox": [min_lon, min_lat, max_lon, max_lat],
            "known_length_count": len(known_lengths),
            "missing_length_count": len(links) - len(known_lengths),
            "computed_length_count": computed_length_count,
            "total_length_km": sum(known_lengths) / 1000.0,
            "missing_endpoint_count": sum(
                1 for link in links if link["from_id"] not in node_ids or link["to_id"] not in node_ids
            ),
        },
    }


def render_outputs(network: dict[str, Any], html_path: Path, svg_path: Path) -> None:
    render_data = _project_network(network)
    svg_markup = _build_svg(render_data, standalone=False)
    svg_path.write_text(_standalone_svg(render_data), encoding="utf-8")
    html_path.write_text(_html_document(network, render_data, svg_markup), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the road_map GeoJSON network to HTML and SVG.")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES_PATH)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS_PATH)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG_PATH)
    parser.add_argument("--station-setting", type=Path, default=DEFAULT_STATION_SETTINGS_PATH)
    args = parser.parse_args()

    network = load_network(args.nodes, args.links, load_renewable_station_ids(args.station_setting))
    render_outputs(network, args.html, args.svg)
    stats = network["stats"]
    print(f"nodes: {stats['node_count']}")
    print(f"links: {stats['link_count']}")
    print(f"node types: {stats['node_type_counts']}")
    print(f"known link length: {stats['known_length_count']}; missing: {stats['missing_length_count']}")
    print(f"total known length km: {stats['total_length_km']:.2f}")
    print(f"html: {args.html}")
    print(f"svg: {args.svg}")


def _zero_snap(value: float) -> float:
    return 0.0 if abs(value) < 1e-14 else value


def _read_feature_collection(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection.")
    return data


def _is_xy(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _iter_xy(coordinates: Any) -> Any:
    if _is_xy(coordinates):
        yield coordinates[0], coordinates[1]
        return
    if isinstance(coordinates, list):
        for child in coordinates:
            yield from _iter_xy(child)


def _format_length(length: Any) -> str:
    if not isinstance(length, (int, float)) or isinstance(length, bool):
        return "unknown"
    if length >= 1000:
        return f"{length / 1000:.2f} km"
    return f"{length:.0f} m"


def _polyline_length(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _project_network(network: dict[str, Any]) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = network["stats"]["mercator_bbox"]
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    width = 1300
    margin = 56
    plot_width = width - margin * 2
    plot_height = int(plot_width * span_y / span_x)
    height = max(720, plot_height + margin * 2)

    def project(x: float, y: float) -> tuple[float, float]:
        sx = margin + (x - min_x) / span_x * plot_width
        sy = margin + (max_y - y) / span_y * plot_height
        return sx, sy

    nodes = []
    for node in network["nodes"]:
        sx, sy = project(node["x"], node["y"])
        nodes.append({**node, "sx": sx, "sy": sy, "class_type": _class_type(node["type"])})

    links = []
    for index, link in enumerate(network["links"]):
        projected_points = [project(x, y) for x, y in link["points"]]
        links.append({**link, "index": index + 1, "projected_points": projected_points})

    return {
        "width": width,
        "height": height,
        "view_box": f"0 0 {width} {height}",
        "nodes": nodes,
        "links": links,
        "stats": network["stats"],
    }


def _class_type(node_type: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", str(node_type).lower()).strip("-")
    return value or "unknown"


def _build_svg(render_data: dict[str, Any], standalone: bool) -> str:
    parts = [
        f'<svg id="network-svg" class="network-svg" viewBox="{render_data["view_box"]}" '
        f'role="img" aria-label="Road map network with {render_data["stats"]["node_count"]} nodes '
        f'and {render_data["stats"]["link_count"]} links" xmlns="http://www.w3.org/2000/svg">',
        f'<rect class="background" width="{render_data["width"]}" height="{render_data["height"]}" />',
        '<g class="links-layer">',
    ]
    for link in render_data["links"]:
        path_d = _path_d(link["projected_points"])
        label = (
            f'{link["from_id"]} -> {link["to_id"]}; length {link["length_text"]}; '
            f'source {link["length_source"]}'
        )
        parts.append(
            '<path class="link" '
            f'd="{path_d}" '
            f'data-kind="link" data-index="{link["index"]}" '
            f'data-from="{_attr(link["from_id"])}" data-to="{_attr(link["to_id"])}" '
            f'data-length="{_attr(link["length_text"])}" data-length-source="{_attr(link["length_source"])}">'
            f"<title>{escape(label)}</title></path>"
        )
    parts.append("</g>")

    ordered_types = ["toll", "interchange", "service"]
    unknown_types = sorted(
        node_type for node_type in {node["type"] for node in render_data["nodes"]} if node_type not in ordered_types
    )
    for node_type in ordered_types + unknown_types:
        class_type = _class_type(node_type)
        parts.append(f'<g class="nodes-layer node-layer-{class_type}">')
        for node in (node for node in render_data["nodes"] if node["type"] == node_type):
            style = NODE_STYLES.get(node_type, NODE_STYLES["unknown"])
            renewable_class = " node-renewable" if node["renewable"] else ""
            renewable_label = "; renewable" if node["renewable"] else ""
            title = f'Node {node["id"]}; {node["type"]}{renewable_label}; lon {node["lon"]:.6f}; lat {node["lat"]:.6f}'
            parts.append(
                '<circle class="node '
                f'node-{class_type}{renewable_class}" '
                f'cx="{node["sx"]:.2f}" cy="{node["sy"]:.2f}" r="{style["radius"]}" '
                f'fill="{style["color"]}" '
                f'data-kind="node" data-id="{_attr(node["id"])}" data-type="{_attr(node["type"])}" '
                f'data-renewable="{str(node["renewable"]).lower()}" '
                f'data-lon="{node["lon"]:.6f}" data-lat="{node["lat"]:.6f}" '
                f'data-x="{node["sx"]:.2f}" data-y="{node["sy"]:.2f}">'
                f"<title>{escape(title)}</title></circle>"
            )
        parts.append("</g>")

    parts.append('<g class="service-labels">')
    for node in (node for node in render_data["nodes"] if node["type"] == "service"):
        parts.append(
            f'<text class="service-label" x="{node["sx"] + 7:.2f}" y="{node["sy"] - 7:.2f}">'
            f'{escape(str(node["id"]))}</text>'
        )
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def _standalone_svg(render_data: dict[str, Any]) -> str:
    style = """
<style>
  .background { fill: #f7fafc; }
  .link { fill: none; stroke: #526d7a; stroke-width: 1.15; stroke-opacity: 0.58; vector-effect: non-scaling-stroke; }
  .node { stroke: #ffffff; stroke-width: 1.3; stroke-opacity: 0.92; vector-effect: non-scaling-stroke; }
  .service-label { display: none; font: 11px Arial, sans-serif; fill: #1f2933; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; }
</style>
""".strip()
    svg = _build_svg(render_data, standalone=True)
    return svg.replace(">", f">\n{style}", 1)


def _html_document(network: dict[str, Any], render_data: dict[str, Any], svg_markup: str) -> str:
    stats = network["stats"]
    node_type_counts = stats["node_type_counts"]
    renewable_station_count = stats["renewable_station_count"]
    min_lon, min_lat, max_lon, max_lat = stats["lonlat_bbox"]
    layers = "\n".join(
        _layer_checkbox(node_type, node_type_counts.get(node_type, 0))
        for node_type in ["toll", "interchange", "service"]
        if node_type_counts.get(node_type, 0)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Road Map Visualization</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #617080;
      --panel: #ffffff;
      --line: #d7e0e7;
      --map-bg: #eef4f7;
      --accent: #2364aa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: #e8eef2;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(280px, 336px) minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      display: flex;
      flex-direction: column;
      gap: 18px;
      padding: 22px;
      background: var(--panel);
      border-right: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.18;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdfe;
    }}
    .metric strong {{
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }}
    .metric span {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .section {{
      padding-top: 2px;
    }}
    .layer {{
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 7px 0;
      color: #24313d;
      font-size: 14px;
    }}
    .layer input {{
      width: 16px;
      height: 16px;
      margin: 0;
      accent-color: var(--accent);
    }}
    .swatch {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      border: 1px solid #ffffff;
      box-shadow: 0 0 0 1px rgba(23, 33, 43, .16);
      flex: 0 0 auto;
    }}
    .layer-count {{
      margin-left: auto;
      color: var(--muted);
      font-size: 12px;
    }}
    .search-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
    }}
    input[type="search"] {{
      min-width: 0;
      height: 34px;
      border: 1px solid #c8d4dd;
      border-radius: 7px;
      padding: 0 10px;
      color: var(--ink);
      font: inherit;
    }}
    button {{
      height: 34px;
      border: 1px solid #b8c7d3;
      border-radius: 7px;
      padding: 0 11px;
      color: #183145;
      background: #f8fbfd;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ background: #eef5f9; }}
    .details {{
      min-height: 136px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdfe;
    }}
    dl {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 7px 12px;
      margin: 0;
      font-size: 13px;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }}
    .bounds {{
      margin-top: auto;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .map-shell {{
      display: flex;
      min-width: 0;
      min-height: 100vh;
      padding: 16px;
    }}
    .canvas-wrap {{
      width: 100%;
      min-height: calc(100vh - 32px);
      border: 1px solid #cbd8e0;
      border-radius: 8px;
      overflow: hidden;
      background: var(--map-bg);
    }}
    .network-svg {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: calc(100vh - 34px);
      cursor: grab;
      touch-action: none;
      background: #f7fafc;
    }}
    .network-svg.is-panning {{ cursor: grabbing; }}
    .background {{ fill: #f7fafc; }}
    .link {{
      fill: none;
      stroke: #516f7d;
      stroke-width: 1.25;
      stroke-opacity: .58;
      vector-effect: non-scaling-stroke;
      cursor: pointer;
    }}
    .link:hover, .link.selected {{
      stroke: #111827;
      stroke-width: 3.2;
      stroke-opacity: .92;
    }}
    .node {{
      stroke: #ffffff;
      stroke-width: 1.35;
      stroke-opacity: .94;
      vector-effect: non-scaling-stroke;
      cursor: pointer;
    }}
    .node:hover, .node.selected {{
      stroke: #111827;
      stroke-width: 3;
    }}
    .service-label {{
      display: none;
      font: 11px Arial, sans-serif;
      fill: #1f2933;
      pointer-events: none;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 3px;
    }}
    body.show-service-labels .service-label {{ display: block; }}
    body.hide-toll .node-toll,
    body.hide-interchange .node-interchange,
    body.hide-service .node-service,
    body.hide-service .service-label {{ display: none; }}
    body.highlight-renewable .node-renewable {{ fill: #f59e0b !important; }}
    @media (max-width: 860px) {{
      .app {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .map-shell {{
        min-height: 68vh;
      }}
      .canvas-wrap, .network-svg {{
        min-height: 68vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <header>
        <h1>Road Map</h1>
      </header>
      <section class="metric-grid" aria-label="Network summary">
        <div class="metric"><strong>{stats["node_count"]}</strong><span>Nodes</span></div>
        <div class="metric"><strong>{stats["link_count"]}</strong><span>Links</span></div>
        <div class="metric"><strong>{node_type_counts.get("service", 0)}</strong><span>Services</span></div>
        <div class="metric"><strong>{stats["total_length_km"]:.0f}</strong><span>Known km</span></div>
      </section>
      <section class="section">
        <h2>Layers</h2>
        {layers}
        <label class="layer">
          <input id="renewable-toggle" type="checkbox">
          <span class="swatch" style="background:#f59e0b"></span>
          <span>Renewable stations</span>
          <span class="layer-count">{renewable_station_count}</span>
        </label>
        <label class="layer">
          <input id="service-label-toggle" type="checkbox">
          <span>Service IDs</span>
        </label>
      </section>
      <section class="section">
        <h2>Find</h2>
        <div class="search-row">
          <input id="node-search" type="search" inputmode="numeric" placeholder="Node id">
          <button id="fit-button" type="button" title="Fit view">Fit</button>
          <button id="clear-button" type="button" title="Clear selection">Clear</button>
        </div>
      </section>
      <section class="details">
        <h2>Selection</h2>
        <dl id="details"></dl>
      </section>
      <footer class="bounds">
        lon {min_lon:.4f} to {max_lon:.4f}<br>
        lat {min_lat:.4f} to {max_lat:.4f}<br>
        EPSG:3857 converted to WGS84
      </footer>
    </aside>
    <main class="map-shell">
      <div class="canvas-wrap">
        {svg_markup}
      </div>
    </main>
  </div>
  <script>
    const svg = document.getElementById('network-svg');
    let viewBox = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
    const initialViewBox = viewBox.slice();
    const details = document.getElementById('details');
    const search = document.getElementById('node-search');
    const nodesById = new Map([...document.querySelectorAll('.node')].map((node) => [node.dataset.id, node]));

    function setViewBox(next) {{
      viewBox = next;
      svg.setAttribute('viewBox', next.map((value) => value.toFixed(2)).join(' '));
    }}

    function svgPoint(event) {{
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      return point.matrixTransform(svg.getScreenCTM().inverse());
    }}

    function setDetails(rows) {{
      details.textContent = '';
      for (const [label, value] of rows) {{
        const term = document.createElement('dt');
        const desc = document.createElement('dd');
        term.textContent = label;
        desc.textContent = value;
        details.append(term, desc);
      }}
    }}

    function clearSelection() {{
      document.querySelectorAll('.selected').forEach((item) => item.classList.remove('selected'));
      setDetails([]);
    }}

    function selectElement(element, zoomTo = false) {{
      clearSelection();
      element.classList.add('selected');
      if (element.dataset.kind === 'node') {{
        setDetails([
          ['Kind', 'node'],
          ['ID', element.dataset.id],
          ['Type', element.dataset.type],
          ['Renewable', element.dataset.renewable],
          ['Lon', element.dataset.lon],
          ['Lat', element.dataset.lat],
        ]);
        if (zoomTo) {{
          const cx = Number(element.dataset.x);
          const cy = Number(element.dataset.y);
          const size = Math.min(initialViewBox[2], initialViewBox[3]) * 0.18;
          setViewBox([cx - size / 2, cy - size / 2, size, size]);
        }}
      }} else {{
        setDetails([
          ['Kind', 'link'],
          ['Index', element.dataset.index],
          ['From', element.dataset.from],
          ['To', element.dataset.to],
          ['Length', element.dataset.length],
          ['Length source', element.dataset.lengthSource],
        ]);
      }}
    }}

    document.querySelectorAll('.node, .link').forEach((element) => {{
      element.addEventListener('click', (event) => {{
        event.stopPropagation();
        selectElement(element);
      }});
    }});

    document.querySelectorAll('[data-toggle-type]').forEach((input) => {{
      input.addEventListener('change', () => {{
        document.body.classList.toggle(`hide-${{input.dataset.toggleType}}`, !input.checked);
      }});
    }});

    document.getElementById('service-label-toggle').addEventListener('change', (event) => {{
      document.body.classList.toggle('show-service-labels', event.target.checked);
    }});

    document.getElementById('renewable-toggle').addEventListener('change', (event) => {{
      document.body.classList.toggle('highlight-renewable', event.target.checked);
    }});

    search.addEventListener('change', () => {{
      const node = nodesById.get(search.value.trim());
      if (node) selectElement(node, true);
    }});

    document.getElementById('fit-button').addEventListener('click', () => {{
      setViewBox(initialViewBox.slice());
    }});

    document.getElementById('clear-button').addEventListener('click', () => {{
      search.value = '';
      clearSelection();
    }});

    svg.addEventListener('wheel', (event) => {{
      event.preventDefault();
      const pointer = svgPoint(event);
      const factor = event.deltaY > 0 ? 1.12 : 0.88;
      setViewBox([
        pointer.x - (pointer.x - viewBox[0]) * factor,
        pointer.y - (pointer.y - viewBox[1]) * factor,
        viewBox[2] * factor,
        viewBox[3] * factor,
      ]);
    }}, {{ passive: false }});

    let panStart = null;
    let panViewBox = null;
    svg.addEventListener('pointerdown', (event) => {{
      if (event.button !== 0) return;
      panStart = {{ x: event.clientX, y: event.clientY }};
      panViewBox = viewBox.slice();
      svg.classList.add('is-panning');
      svg.setPointerCapture(event.pointerId);
    }});
    svg.addEventListener('pointermove', (event) => {{
      if (!panStart) return;
      const dx = (event.clientX - panStart.x) * panViewBox[2] / svg.clientWidth;
      const dy = (event.clientY - panStart.y) * panViewBox[3] / svg.clientHeight;
      setViewBox([panViewBox[0] - dx, panViewBox[1] - dy, panViewBox[2], panViewBox[3]]);
    }});
    svg.addEventListener('pointerup', (event) => {{
      panStart = null;
      panViewBox = null;
      svg.classList.remove('is-panning');
      svg.releasePointerCapture(event.pointerId);
    }});
  </script>
</body>
</html>
"""


def _layer_checkbox(node_type: str, count: int) -> str:
    style = NODE_STYLES.get(node_type, NODE_STYLES["unknown"])
    class_type = _class_type(node_type)
    return (
        '<label class="layer">'
        f'<input type="checkbox" checked data-toggle-type="{class_type}">'
        f'<span class="swatch" style="background:{style["color"]}"></span>'
        f'<span>{escape(style["label"])}</span>'
        f'<span class="layer-count">{count}</span>'
        '</label>'
    )


def _path_d(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first_x, first_y = points[0]
    commands = [f"M {first_x:.2f} {first_y:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(commands)


def _attr(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


if __name__ == "__main__":
    main()
