"""
Topology Bubble Planner
-----------------------

Program-driven mode:
   - Read the previous step's building_program.json (reasoning + floors[])
   - For each floor, build a floor prompt from program data and the corresponding
     floor polygon JSON produced after core cut-out (floor1/others), and
     optionally attach per-floor PNGs. Floors are planned concurrently.
   - Output per-floor topology JSONs and bubble graphs.
"""
from __future__ import annotations  # moved into core package

import argparse
import base64
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mansion.llm.openai_wrapper import OpenAIWrapper

try:
    from shapely.geometry import Polygon  # type: ignore
except Exception:  # noqa: BLE001
    Polygon = None  # type: ignore


class TopologyBubblePlanner:
    def __init__(self, output_dir: Optional[str] = None, workers: Optional[int] = None, llm: Optional[OpenAIWrapper] = None) -> None:
        if llm:
            self.llm = llm
        else:
            self.llm = OpenAIWrapper()
            
        self.output_dir = os.path.join(output_dir or "llm_planning_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.default_floor_design: Optional[str] = None
        self.default_wall_design: Optional[str] = None
        # concurrency (for per-floor LLM calls)
        self.workers: Optional[int] = workers

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def ask_llm(self, prompt: str, img_path: Optional[str] = None, raw_prefix: str = "topology_raw") -> Optional[Dict[str, Any]]:
        from mansion.generation import prompts
        
        messages: List[Dict[str, Any]]
        if img_path and os.path.exists(img_path):
            image_encoded = self._encode_image(img_path)
            messages = [
                {"role": "system", "content": prompts.TOPOLOGY_PLANNER_TEMPLATE},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_encoded}"}}
                ]},
            ]
        else:
            messages = [
                {"role": "system", "content": prompts.TOPOLOGY_PLANNER_TEMPLATE},
                {"role": "user", "content": prompt},
            ]

        try:
            text = self.llm.chat(
                messages,
                temperature=0.2,
                max_tokens=20000,
            )
            # Save raw assistant text for inspection
            try:
                raw_path = os.path.join(self.output_dir, f"{raw_prefix}.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

            jt = text.strip()
            if jt.startswith("```"):
                if jt.startswith("```json"):
                    jt = jt[len("```json"):].strip()
                else:
                    jt = jt[len("```"):].strip()
                if jt.endswith("```"):
                    jt = jt[:-3].strip()
            # Also save stripped JSON text for debugging
            try:
                stripped_path = os.path.join(self.output_dir, f"{raw_prefix}_stripped.json")
                with open(stripped_path, "w", encoding="utf-8") as f:
                    f.write(jt)
            except Exception:
                pass
            def _try_load(s: str) -> Optional[Dict[str, Any]]:
                try:
                    return json.loads(s)
                except Exception:
                    return None

            data = _try_load(jt)
            if data is None:
                # Fallback: trim to the last closing brace and strip trailing commas
                import re
                # take substring between first '{' and last '}'
                b0 = jt.find('{')
                b1 = jt.rfind('}')
                if b0 != -1 and b1 != -1 and b1 > b0:
                    cand = jt[b0:b1+1]
                    # remove trailing commas like ,}\n or ,]\n
                    cand = re.sub(r',\s*([}\]])', r'\1', cand)
                    data = _try_load(cand)
                if data is None:
                    print("[ERROR] Topology generation failed: JSON parsing failed (repair/cropping already attempted)")
                    return None
            return data
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Topology generation failed: {exc}")
            return None

    # -------------------- Visualization: Bubble Graph --------------------
    @staticmethod
    def _pick_label(node: Dict[str, Any]) -> str:
        # New rule: labels derive from id only (name removed)
        idv = str(node.get("id", ""))
        return idv if idv else "node"

    @staticmethod
    def _find_main_id(nodes: List[Dict[str, Any]]) -> str:
        for n in nodes:
            if str(n.get("type", "")).lower() == "main":
                return n.get("id")
        for n in nodes:
            if str(n.get("id", "")).lower() == "main":
                return n.get("id")
        return nodes[0].get("id") if nodes else "main"

    def _build_positions(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        # Build undirected adjacency using allowed kinds (Regions/transition disabled)
        allowed = {"access", "adjacent"}
        adj: Dict[str, List[str]] = {}
        ekind: Dict[Tuple[str, str], str] = {}
        for e in edges:
            s = e.get("source"); t = e.get("target")
            k = str(e.get("kind", "")).lower()
            # Normalize legacy keys: 'acces' -> 'access', 'internal' -> 'adjacent'
            if k == "acces":
                k = "access"
            if k == "internal":
                k = "adjacent"
            if not s or not t:
                continue
            if k not in allowed:
                k = "adjacent"
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, []).append(s)
            ekind[(s, t)] = k
            ekind[(t, s)] = k

        main_id = self._find_main_id(nodes)

        # BFS levels from main (layered rings)
        from collections import deque, defaultdict
        level: Dict[str, int] = {main_id: 0}
        parent: Dict[str, str] = {}
        q: deque[str] = deque([main_id])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                cand = level[u] + 1
                if v not in level or cand < level[v]:
                    level[v] = cand
                    parent[v] = u
                    q.append(v)

        # Group by levels
        by_level: Dict[int, List[str]] = defaultdict(list)
        for nid, lv in level.items():
            by_level[lv].append(nid)

        # Base radii and angle maps
        R1 = 5.0
        R_step = 3.0
        angle: Dict[str, float] = {main_id: 0.0}
        pos: Dict[str, Tuple[float, float]] = {main_id: (0.0, 0.0)}

        # Ring 1: distribute evenly
        ring1 = [n for n in sorted(by_level.get(1, [])) if n != main_id]
        if ring1:
            step = 2 * math.pi / max(1, len(ring1))
            for i, nid in enumerate(ring1):
                ang = i * step
                angle[nid] = ang
                pos[nid] = (R1 * math.cos(ang), R1 * math.sin(ang))

        # Outer rings: align with parent direction, fan out children
        max_level = max(by_level.keys()) if by_level else 0
        for lv in range(2, max_level + 1):
            ring_nodes = [n for n in sorted(by_level.get(lv, [])) if n != main_id]
            # Map parent -> children list for this ring
            children_by_parent: Dict[str, List[str]] = defaultdict(list)
            for nid in ring_nodes:
                p = parent.get(nid)
                if not p:
                    continue
                children_by_parent[p].append(nid)
            for p, kids in children_by_parent.items():
                base_ang = angle.get(p, 0.0)
                m = len(kids)
                if m == 1:
                    offsets = [0.0]
                else:
                    # total spread up to ~60 degrees, scaled by kids count
                    spread = min(math.pi / 3, 0.3 + 0.15 * (m - 1))
                    step_ang = spread / (m - 1)
                    offsets = [(-spread / 2) + j * step_ang for j in range(m)]
                for kid, off in zip(kids, offsets):
                    ang = base_ang + off
                    angle[kid] = ang
                    R = R1 + (lv - 1) * R_step
                    pos[kid] = (R * math.cos(ang), R * math.sin(ang))

            # Place any nodes without known parent angle evenly around the ring
            unplaced = [nid for nid in ring_nodes if nid not in pos]
            if unplaced:
                step = 2 * math.pi / len(unplaced)
                for i, nid in enumerate(unplaced):
                    ang = i * step
                    angle[nid] = ang
                    R = R1 + (lv - 1) * R_step
                    pos[nid] = (R * math.cos(ang), R * math.sin(ang))

        return pos

    def draw_bubble_graph(self, topo: Dict[str, Any], output_path: str) -> None:
        nodes: List[Dict[str, Any]] = topo.get("nodes", [])
        edges: List[Dict[str, Any]] = topo.get("edges", [])
        if not nodes:
            print("[ERROR] Empty node set; cannot draw bubble graph.")
            return

        pos = self._build_positions(nodes, edges)
        main_id = self._find_main_id(nodes)

        # Styling helpers
        # Precompute area-based scaling for consistent bubble sizes
        area_vals = []
        for n in nodes:
            try:
                if n.get("area") is not None:
                    area_vals.append(float(n.get("area") or 0.0))
            except Exception:
                pass
        amin = min(area_vals) if area_vals else 0.0
        amax = max(area_vals) if area_vals else 1.0
        def scale(a: float, smin: float = 380.0, smax: float = 1100.0) -> float:
            if amax <= amin:
                return (smin + smax) / 2.0
            t = max(0.0, min(1.0, (a - amin) / (amax - amin)))
            return smin + t * (smax - smin)

        def node_style(n: Dict[str, Any]) -> Tuple[str, float, str]:
            ntype = str(n.get("type", "")).lower()
            try:
                area_val = float(n.get("area") or 0.0)
            except Exception:
                area_val = 0.0
            if ntype == "main" or n.get("id") == main_id:
                return ("#e76f51", scale(area_val, 600.0, 1250.0), "black")
            if ntype in ("exit",):
                return ("#6c757d", 480.0, "#333333")  # gray
            if ntype in ("elevator", "stair"):
                return ("#8d99ae", 520.0, "#333333")  # vertical connectors
            # Regions disabled: treat as normal entity if present
            palette = ["#f4a261", "#2a9d8f", "#e9c46a", "#8ab17d", "#ffb3ba", "#cdb4db", "#bde0fe", "#ffd6a5"]
            idx = (hash(n.get("id")) % len(palette))
            return (palette[idx], scale(area_val, 420.0, 980.0), "#333333")

        kind_to_style = {
            "access": dict(color="#444", lw=2.4, ls="-", alpha=0.9),
            "adjacent": dict(color="#666", lw=1.8, ls="-", alpha=0.8),
        }

        fig, ax = plt.subplots(figsize=(6, 6))

        # Edges
        for e in edges:
            s, t = e.get("source"), e.get("target")
            if s not in pos or t not in pos:
                continue
            k = str(e.get("kind", "adjacent")).lower()
            if k == "acces":
                k = "access"
            if k == "internal":
                k = "adjacent"
            if k not in kind_to_style:
                k = "adjacent"
            style = kind_to_style.get(k, kind_to_style["adjacent"])
            xs, ys = pos[s]
            xt, yt = pos[t]
            ax.plot([xs, xt], [ys, yt], color=style["color"], linewidth=style["lw"], linestyle=style["ls"], alpha=style["alpha"])  # noqa: E501

        # Nodes + labels (include area if provided)
        for n in nodes:
            nid = n.get("id")
            if nid not in pos:
                continue
            x, y = pos[nid]
            face, size, edge = node_style(n)
            ax.scatter([x], [y], s=size, c=face, edgecolors=edge, linewidths=1.2, zorder=3)
            label = self._pick_label(n)
            try:
                if "area" in n and n["area"] is not None:
                    a = float(n["area"])
                    label = f"{label}\n{a:.1f} m²"
            except Exception:
                pass
            ax.text(x, y, label, ha='center', va='center', fontsize=9, color='black')

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        margin = 3.0
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()
        # Build edge legend in English
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], color=kind_to_style["access"]["color"], lw=kind_to_style["access"]["lw"], ls=kind_to_style["access"]["ls"], label="Access"),
            Line2D([0], [0], color=kind_to_style["adjacent"]["color"], lw=kind_to_style["adjacent"]["lw"], ls=kind_to_style["adjacent"]["ls"], label="Adjacent"),
        ]
        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)

        ax.set_title('Topological Bubble Graph', fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f"[VIS] Bubble graph saved: {output_path}")

    # ---------- Area post-processing ----------
    def _enforce_area_rules(self, topo: Dict[str, Any], gfa: Optional[float] = None) -> Dict[str, Any]:
        nodes = topo.get("nodes", []) or []
        edges = topo.get("edges", []) or []
        id2n = {n.get("id"): n for n in nodes if n.get("id")}
        # normalize area
        for n in nodes:
            try:
                if "area" in n and n["area"] is not None:
                    n["area"] = float(n["area"])  # type: ignore[assignment]
            except Exception:
                n["area"] = None  # type: ignore[index]

        # Build region children via transition
        def is_region(nid: str) -> bool:
            t = str(id2n.get(nid, {}).get("type", "")).lower()
            return t in ("regions", "region")

        children_by_region: Dict[str, List[str]] = {}
        for e in edges:
            k = str(e.get("kind", "")).lower()
            if k == "internal":
                k = "transition"
            if k != "transition":
                continue
            s, t = e.get("source"), e.get("target")
            if s in id2n and is_region(s) and t in id2n:
                children_by_region.setdefault(s, []).append(t)
            elif t in id2n and is_region(t) and s in id2n:
                children_by_region.setdefault(t, []).append(s)

        # Enforce region partition: sum(children)==region.area
        for rid, kids in children_by_region.items():
            r = id2n.get(rid)
            if not r:
                continue
            try:
                r_area = float(r.get("area") or 0.0)
            except Exception:
                r_area = 0.0
            if r_area <= 0 or not kids:
                continue
            # get kids areas
            vals = []
            for kid in kids:
                kv = id2n.get(kid)
                a = None
                try:
                    a = float(kv.get("area")) if kv and kv.get("area") is not None else None
                except Exception:
                    a = None
                vals.append(a)
            if any(v is None for v in vals):
                share = r_area / len(kids)
                for kid in kids:
                    if kid in id2n:
                        id2n[kid]["area"] = round(share, 1)
            else:
                s = sum(v for v in vals if v is not None)
                if s <= 0:
                    share = r_area / len(kids)
                    for kid in kids:
                        id2n[kid]["area"] = round(share, 1)
                else:
                    scale = r_area / s
                    for kid, v in zip(kids, vals):
                        id2n[kid]["area"] = round(float(v) * scale, 1)

        # If GFA provided, set main = GFA - sum(other areas, while avoiding double-counting
        # conceptual/aggregating nodes such as Regions/area and vertical cores).
        if gfa is not None and gfa > 0:
            tot_entities = 0.0
            for n in nodes:
                t = str(n.get("type", "")).lower()
                # Regions/region: purely aggregating, do not subtract twice
                if t in ("regions", "region"):
                    continue
                # area: Functional partition node is just a semantic grouping of several physical rooms and will not appear as an independent room itself.
                # Area is usually already reflected in its sub-entities; here it is no longer included in the total to avoid double deductions for area.
                if t == "area":
                    continue
                # Vertical traffic nodes (stairs, elevators, etc.) usually already occupy a separate space in the geometric outline and should no longer be deducted from the area of ​​​​main.
                if t in ("stair", "elevator") or "stair" in t or "elevator" in t or "lift" in t or "core" in t:
                    continue
                if t == "main":
                    continue
                try:
                    a = float(n.get("area") or 0.0)
                except Exception:
                    a = 0.0
                tot_entities += a
            main_node = next((n for n in nodes if str(n.get("type", "")).lower() == "main"), None)
            if main_node is not None:
                main_node["area"] = round(max(0.0, gfa - tot_entities), 1)

        topo["nodes"] = nodes
        topo["edges"] = edges
        return topo

    # ------- Helpers: vertical elements from layout + warnings -------
    def _vertical_hints_from_layout(self, layout: Dict[str, Any]) -> Dict[str, Any]:
        nodes = layout.get("nodes", {}) or {}
        hints: Dict[str, Any] = {"stair": [], "elevator": []}
        for name, node in nodes.items():
            if not isinstance(name, str):
                continue
            base = name.split("_")[0].lower()
            if base in ("stair", "elevator"):
                coords = node.get("polygon") or []
                if len(coords) >= 3:
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    hints[base].append({"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]})
        return hints

    def _warn_missing_verticals(self, topo: Dict[str, Any], vertical_hints: Dict[str, Any], is_floor1: bool = False) -> None:
        nodes = topo.get("nodes", []) or []
        types = [str(n.get("type", "")).lower() for n in nodes]
        if vertical_hints.get("stair") and "stair" not in types:
            print("[WARN] Stair core exists on this floor, but topology output has no node with type='stair'.")
        if vertical_hints.get("elevator") and "elevator" not in types:
            print("[WARN] Elevator core exists on this floor, but topology output has no node with type='elevator'.")

    def _inject_missing_verticals(self, topo: Dict[str, Any], vertical_hints: Dict[str, Any]) -> Dict[str, Any]:
        """If layout has elevator/stair but topo missed them, inject minimal nodes (+access edge to main)."""
        nodes = topo.get("nodes", []) or []
        edges = topo.get("edges", []) or []
        types = [str(n.get("type", "")).lower() for n in nodes]
        mains = [n for n in nodes if str(n.get("type", "")).lower() == "main"]
        main_id = mains[0].get("id") if len(mains) == 1 else None

        def _add_node(nid: str, ntype: str, bbox: Dict[str, Any]) -> None:
            xs = bbox.get("x") or []
            ys = bbox.get("y") or []
            if len(xs) == 2 and len(ys) == 2:
                try:
                    area = abs(float(xs[1] - xs[0]) * float(ys[1] - ys[0]))
                except Exception:
                    area = 0.0
            else:
                area = 0.0
            nodes.append({
                "id": nid,
                "type": ntype,
                "area": area or 4.0,
                "floor_material": None,
                "wall_material": None,
                "open_relation": "door",
            })
            if main_id:
                edges.append({"source": nid, "target": main_id, "kind": "access"})

        if vertical_hints.get("elevator") and "elevator" not in types:
            bbox = (vertical_hints.get("elevator") or [{}])[0]
            _add_node("elevator", "elevator", bbox)
        if vertical_hints.get("stair") and "stair" not in types:
            bbox = (vertical_hints.get("stair") or [{}])[0]
            _add_node("stair", "stair", bbox)
        topo["nodes"] = nodes
        topo["edges"] = edges
        return topo

    def _ensure_access_edges(self, topo: Dict[str, Any], require_types: Tuple[str, ...] = ("elevator", "stair")) -> Dict[str, Any]:
        """Suggest missing access connections for vertical nodes; do NOT mutate JSON."""
        nodes = topo.get("nodes", []) or []
        edges = topo.get("edges", []) or []
        mains = [n for n in nodes if str(n.get("type", "")).lower() == "main"]
        if len(mains) != 1:
            return topo
        main_id = mains[0].get("id")
        if not main_id:
            return topo
        # quick lookup
        def norm_kind(k: Any) -> str:
            k = str(k or "").lower()
            return "access" if k == "acces" else k
        existing = set()
        for e in edges:
            s = e.get("source"); t = e.get("target"); k = norm_kind(e.get("kind"))
            if s and t:
                existing.add(tuple(sorted((s, t))) + (k,))
        # collect targets
        targets: List[str] = []
        for n in nodes:
            t = str(n.get("type", "")).lower()
            if t in require_types and n.get("id"):
                targets.append(n["id"])  # type: ignore[index]
        for nid in targets:
            key = tuple(sorted((nid, main_id))) + ("access",)
            if key not in existing:
                print(f"[INFO] Suggest adding an 'access' edge from vertical node '{nid}' to main '{main_id}'.")
        return topo

    def _sanitize_open_relations(self, topo: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce open_relation policy on nodes in-place:
        - Only allow 'open' or 'door'
        - main -> 'open' (enforced)
        - elevator -> 'door' (enforced)
        - stair -> 'door' (enforced)
        - other nodes: keep LLM output if valid ('open'/'door'), else default to 'door'
        """
        nodes = topo.get("nodes", []) or []
        for n in nodes:
            t = str(n.get("type", "")).lower().strip()
            v = str(n.get("open_relation", "")).lower().strip()
            
            # Mandatory constraints: main, elevator, stair
            if t == "main":
                n["open_relation"] = "open"
                continue
            if t == "elevator":
                n["open_relation"] = "door"
                continue
            if t == "stair":
                n["open_relation"] = "door"
                continue
            
            # Other nodes: retain the output of LLM and only verify the validity
            if v not in ("open", "door"):
                # If LLM outputs an invalid value, defaults to 'door'
                n["open_relation"] = "door"
            else:
                # Preserve valid output from LLM
                n["open_relation"] = v
        topo["nodes"] = nodes
        return topo

    def _auto_select_main_by_area(self, topo: Dict[str, Any], gfa: Optional[float] = None, prefer_id: Optional[str] = None, ratio: float = 0.35, margin: float = 1.25) -> Dict[str, Any]:
        """Suggest a main node based on area, but do not modify the JSON.

        Prints hints to the terminal when:
        - No main is present and a good candidate exists (prefer program's first room id, else largest by area)
        - A non-main node strongly dominates the area and is likely the true hub
        - Multiple mains are present
        """
        nodes = topo.get("nodes", []) or []
        if not nodes:
            return topo

        def area_of(n):
            try:
                return float(n.get("area") or 0.0)
            except Exception:
                return 0.0

        mains = [n for n in nodes if str(n.get("type", "")).lower() == "main"]
        largest = max(nodes, key=area_of)
        largest_area = area_of(largest)
        gfa_val = float(gfa) if gfa else None

        if len(mains) == 0:
            if prefer_id and any(n.get("id") == prefer_id for n in nodes):
                print(f"[INFO] No main node detected. Suggest marking '{prefer_id}' as main (first room from program).")
            else:
                print(f"[INFO] No main node detected. Suggest selecting '{largest.get('id')}' as main by area (area≈{largest_area:.1f}).")
            return topo

        if len(mains) > 1:
            ids = ", ".join(n.get("id", "?") for n in mains)
            print(f"[INFO] Multiple main nodes detected: {ids}. Suggest keeping only one main node to simplify connectivity.")
            return topo

        current_main = mains[0]
        main_area = area_of(current_main)
        should_promote = False
        if gfa_val and gfa_val > 0 and largest_area / gfa_val >= ratio and largest is not current_main:
            should_promote = True
        elif largest is not current_main and main_area > 0 and largest_area > main_area * margin:
            should_promote = True

        if should_promote:
            print(
                f"[INFO] Area hint: node '{largest.get('id')}' (≈{largest_area:.1f}) is much larger than current main '{current_main.get('id')}' (≈{main_area:.1f}). "
                "If you want a more intuitive traffic hub, consider changing it to type='main'."
            )
        return topo

    def _build_topology_prompt(
        self,
        program: Dict[str, Any],
        floor: Dict[str, Any],
        floor_layout_json: Dict[str, Any],
        vertical_hints: Optional[Dict[str, Any]] = None,
        user_requirement: Optional[str] = None,
    ) -> str:
        from mansion.generation import prompts
        
        idx = floor.get("index")
        requirement = floor.get("requirement") or floor.get("program_requirement") or ""
        gfa = floor.get("gross_floor_area", 0.0)
        rooms = floor.get("rooms", [])
        layout_json = json.dumps(floor_layout_json, indent=2, ensure_ascii=False)
        
        vtext = ""
        if vertical_hints:
            lines = ["Fixed vertical elements from layout:"]
            if vertical_hints.get("stair"):
                lines.append(f"- stair core bboxes: {vertical_hints.get('stair')}")
            if vertical_hints.get("elevator"):
                lines.append(f"- elevator core bboxes: {vertical_hints.get('elevator')}")
            vtext = "\n" + "\n".join(lines) + "\n"

        # Extract material hints from building_program rooms
        material_hints_text = ""
        material_hints = {}
        for room in rooms:
            if isinstance(room, dict):
                room_id = room.get("id")
                if room_id:
                    material_hints[str(room_id)] = {
                        "floor_material": room.get("floor_material"),
                        "wall_material": room.get("wall_material")
                    }
        
        if material_hints:
            material_hints_text = "\nMaterial hints (from building_program):\n"
            for rid, mats in material_hints.items():
                floor_mat = mats.get("floor_material") or "unspecified"
                wall_mat = mats.get("wall_material") or "unspecified"
                material_hints_text += f"- {rid}: floor_material={floor_mat}, wall_material={wall_mat}\n"
            material_hints_text += (
                "Prefer using the material hints above. If a room appears in the list, use its corresponding "
                "materials. If not listed or adjustment is needed, choose suitable materials by room function.\n"
            )

        # User original command part
        user_requirement_text = ""
        if user_requirement:
            user_requirement_text = (
                f"\nOriginal user instruction:\n{user_requirement}\n\n"
                "When designing topology, fully consider this original requirement to ensure room layout and connectivity meet user expectations.\n"
            )

        return prompts.TOPOLOGY_PLANNER_TEMPLATE.format(
            reasoning=program.get("reasoning", ""),
            user_requirement_text=user_requirement_text,
            idx=idx,
            gfa=gfa,
            requirement=requirement,
            layout_json=layout_json,
            vtext=vtext,
            material_hints_text=material_hints_text,
            rooms_json=json.dumps(rooms, ensure_ascii=False, indent=2)
        )

    # ----------------- Program-driven multi-floor planning -----------------
    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def plan_from_program(self, program_json: str, floor1_json: str, others_json: Optional[str] = None,
                           img_floor1: Optional[str] = None, img_others: Optional[str] = None,
                           workers: Optional[int] = None, user_requirement: Optional[str] = None) -> Dict[int, Dict[str, str]]:
        # run sequentially to avoid backend/thread issues

        program = self._load_json(program_json)
        floor1_layout = self._load_json(floor1_json)
        others_layout = self._load_json(others_json) if others_json else None

        floors: List[Dict[str, Any]] = program.get("floors", [])
        if not floors:
            raise ValueError("No floors in program JSON")

        results: Dict[int, Dict[str, str]] = {}

        # Split into base floors (need LLM) and copy floors
        base_floors: List[Dict[str, Any]] = []
        copy_floors: List[Dict[str, Any]] = []
        for fl in floors:
            if "copy" in fl and fl.get("copy") is not None:
                copy_floors.append(fl)
            else:
                base_floors.append(fl)

        def submit_floor(floor: Dict[str, Any]):
            idx = int(floor.get("index", 1))
            layout = floor1_layout if idx == 1 else (others_layout or floor1_layout)
            vertical_hints = self._vertical_hints_from_layout(layout)
            prompt = self._build_topology_prompt(
                program=program,
                floor=floor,
                floor_layout_json=layout,
                vertical_hints=vertical_hints,
                user_requirement=user_requirement,
            )
            img_path = img_floor1 if idx == 1 else img_others
            raw_prefix = f"topology_raw_floor_{idx}"
            topo = self.ask_llm(prompt, img_path, raw_prefix=raw_prefix)
            if topo is None:
                raise RuntimeError(f"LLM failed to produce topology JSON for floor {idx}")
            # Area enforcement using floor layout main area as GFA
            try:
                gfa = float((layout.get("nodes", {}).get("main", {}) or {}).get("area", 0.0))
            except Exception:
                gfa = None
            topo = self._enforce_area_rules(topo, gfa=gfa)
            # preferred hub id from program: first room's id if present
            prefer_id = None
            try:
                rooms = floor.get("rooms", [])
                if isinstance(rooms, list) and rooms:
                    prefer_id = rooms[0].get("id") if isinstance(rooms[0], dict) else None
            except Exception:
                prefer_id = None
            topo = self._auto_select_main_by_area(topo, gfa=gfa, prefer_id=prefer_id)
            self._warn_missing_verticals(topo, vertical_hints, is_floor1=(idx == 1))
            topo = self._inject_missing_verticals(topo, vertical_hints)
            topo = self._ensure_access_edges(topo, require_types=("elevator", "stair"))
            topo = self._sanitize_open_relations(topo)
            if not topo:
                raise RuntimeError(f"LLM failed for floor {idx}")
            out_json = os.path.join(self.output_dir, f"topology_graph_floor_{idx}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(topo, f, ensure_ascii=False, indent=2)
            out_img = os.path.join(self.output_dir, f"topology_graph_floor_{idx}.png")
            self.draw_bubble_graph(topo, out_img)
            return idx, {"json": out_json, "image": out_img}

        # Resolve base floors with LLM (multi-threaded)
        if base_floors:
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                n_workers = int(workers or self.workers or min(4, max(1, len(base_floors))))
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futs = {ex.submit(submit_floor, fl): fl for fl in base_floors}
                    for fut in as_completed(futs):
                        idx, payload = fut.result()
                        results[idx] = payload
            except Exception as exc:
                # Fallback to sequential if threads not available
                print(f"[topology] concurrency fallback due to: {exc}")
                for fl in base_floors:
                    idx, payload = submit_floor(fl)
                    results[idx] = payload

        # Then process copy floors without calling LLM
        for fl in copy_floors:
            idx = int(fl.get("index", 0) or 0)
            ref = fl.get("copy")
            try:
                ref_idx = int(ref)
            except Exception:
                raise ValueError(f"Invalid copy target for floor {idx}: {ref}")
            if ref_idx not in results:
                raise ValueError(f"Copy target floor {ref_idx} has no generated topology; ensure it's not also a copy or run order first.")
            # Copy files
            src = results[ref_idx]
            # Copy JSON with meta update
            src_json = src["json"]
            with open(src_json, "r", encoding="utf-8") as f:
                topo = json.load(f)
            meta = topo.get("meta", {}) if isinstance(topo, dict) else {}
            if isinstance(meta, dict):
                meta.update({"copied_from": ref_idx, "floor": idx})
                topo["meta"] = meta
            out_json = os.path.join(self.output_dir, f"topology_graph_floor_{idx}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(topo, f, ensure_ascii=False, indent=2)
            # Copy image
            import shutil
            out_img = os.path.join(self.output_dir, f"topology_graph_floor_{idx}.png")
            shutil.copyfile(src["image"], out_img)
            results[idx] = {"json": out_json, "image": out_img}

        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topology bubble graph planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Program-driven, multi-floor, concurrent planning
  python -m llm.topology_bubble --program llm/llm_planning_output/building_program.json \
      --floor1 llm/llm_planning_output/floor1_polygon.json \
      --others llm/llm_planning_output/other_floors_polygon.json \
      --image-floor1 llm/llm_planning_output/floor1_polygon.png \
      --image-others llm/llm_planning_output/other_floors_polygon.png \
      --out llm_planning_output

  # Legacy single-floor mode
  python -m llm.topology_bubble llm/building_layout/building_layout.json \
      --requirement "Design this floor's topology" \
      --image llm/building_layout/layout_visualization.png
        """
    )

    # Program-driven args
    parser.add_argument("--program", help="building_program.json from previous step")
    parser.add_argument("--floor1", help="floor1_polygon.json (after cores removed)")
    parser.add_argument("--others", default=None, help="other_floors_polygon.json")
    parser.add_argument("--image-floor1", dest="image_floor1", default=None, help="PNG for floor 1")
    parser.add_argument("--image-others", dest="image_others", default=None, help="PNG for other floors")

    parser.add_argument("--out", default="llm_planning_output", help="Output directory (topology JSON/PNG)")

    args = parser.parse_args()
    planner = TopologyBubblePlanner(output_dir=args.out)
    if not args.program or not args.floor1:
        raise SystemExit("Missing required args. Use --program and --floor1 for program-driven mode.")

    results = planner.plan_from_program(
        program_json=args.program,
        floor1_json=args.floor1,
        others_json=args.others,
        img_floor1=args.image_floor1,
        img_others=args.image_others,
    )
    print("[Outputs]")
    for k in sorted(results.keys()):
        print(f"  floor {k}: {results[k]}")


if __name__ == "__main__":
    main()
