from __future__ import annotations

"""Prompt templates moved out from prompts.py because they are currently unused in portable full pipeline flow."""

floor_plan_prompt = """You are an experienced room designer. Please assist me in crafting a floor plan. Each room is a rectangle. You need to define the four coordinates and specify an appropriate design scheme, including each room's color, material, and texture.
Assume the wall thickness is zero. Please ensure that all rooms are connected, not overlapped, and do not contain each other.
Note: the units for the coordinates are meters.
For example:
living room | maple hardwood, matte | light grey drywall, smooth | [(0, 0), (0, 8), (5, 8), (5, 0)]
kitchen | white hex tile, glossy | light grey drywall, smooth | [(5, 0), (5, 5), (8, 5), (8, 0)]

Here are some guidelines for you:
1. A room's size range (length or width) is 3m to 8m. The maximum area of a room is 48 m^2. Please provide a floor plan within this range and ensure the room is not too small or too large.
2. It is okay to have one room in the floor plan if you think it is reasonable.
3. The room name should be unique.

Now, I need a design for {input}.
Additional requirements: {additional_requirements}.
Your response should be direct and without additional text at the beginning or end."""


wall_height_prompt = """I am now designing {input}. Please help me decide the wall height in meters.
Answer with a number, for example, 3.0. Do not add additional text at the beginning or in the end."""


object_selection_prompt = """Assist me in selecting large, floor-based objects to furnish each room, excluding mats, carpets, and rugs. Provide a comprehensive description since I will use it to retrieve object. If multiple identical items are to be placed in the room, please indicate the quantity.

Present your recommendations in this format: room type | object category | object description | quantity
For example:
living room | sofa | modern sectional, light grey sofa | 1
living room | floor lamp | black, tripod floor lamp | 2
kitchen | fridge | stainless steel, french door refrigerator | 1

Currently, the design in progress is "{input}", featuring these rooms: {rooms}. Please also consider the following additional requirements: {additional_requirements}.

Style coherence rule for large pieces (tables, chairs, sofas, beds, bookshelves, cabinets, wardrobes, dining sets, desks): within the same room, keep style/material/color palette coherent (e.g., modern + light wood + warm neutral fabric). Small accessories can vary freely.

Your response should be precise, without additional text at the beginning or end."""


wall_object_selection_prompt = """Assist me in selecting wall-based objects to furnish each room.
Present your recommendations in this format: room type | object category | object description | quantity
For example:
living room | painting | abstract painting | 2
kitchen | cabinet | white, shaker-style, wall cabinet | 2
bathroom | mirror | rectangular, frameless, wall mirror | 1

Now I want you to design {input} which has these rooms: {rooms}.
Please also consider the following additional requirements: {additional_requirements}.
Your response should be precise, without additional text at the beginning or end."""


ceiling_selection_prompt = """Assist me in selecting ceiling objects (light/fan) to furnish each room.
Present your recommendations in this format: room type | ceiling object description
For example:
living room | modern, 3-light, semi-flush mount ceiling light

Currently, the design in progress is "{input}", featuring these rooms: {rooms}. You need to provide one ceiling object for each room.
Please also consider the following additional requirements: {additional_requirements}.

Your response should be precise, without additional text at the beginning or end. """


small_object_selection_prompt = """As an experienced room designer, you are tasked to bring life into the room by strategically placing more *small* objects. Those objects should only be arranged *on top of* large objects which serve as receptacles. 
The output should be formatted as follows: receptacle | small object-1, quantity, variance type | small object-2, quantity, variance type | ...
Here, the variance type specifies whether the small objects are same or varied. There's no restriction on the number of small objects you can select for each receptacle. An example of this format is as follows:
sofa-0 (living room) | remote control for TV, 1, same | book, 2, varied | gray fabric pillow, 2, varied
tv stand-0 (living room) | 49 inch TV, 1, same | speaker, 2, same

Now, we are designing {input} and the available receptacles in the room include: {receptacles}. Additional requirements for this design project are as follows: {additional_requirements}.
Your response should solely contain the information about the placement of objects and should not include any additional text before or after the main content."""


CUT_PLANNER_TEMPLATE = """You are a floor plan partitioning assistant. The input is an orthogonal polygon for a single floor (with elevator/stair cores already removed), along with the floor's topological bubble diagram (including area estimates for each space).

Your task is to produce a "round-1 cut" plan: output a room list that MUST include main itself (circulation/shared space) and all nodes adjacent to main (graph distance=1), excluding elevator/stair; for each room, provide an approximate bounding box that can accommodate it.

Requirements and semantics:
- This round performs a single unified cut; all items have cut_from fixed as "main".
- Target set = main itself + nodes adjacent to main (graph distance=1), excluding elevator/stair.
- Within the same round, prioritize rooms with larger areas; output order should be from largest to smallest estimated area.
- For each target, provide:
  1) cut_from: always "main"
  2) room_id: the target node id (from the topology graph)
  3) estimate_location: position description (e.g., "upper-left L-shaped wing", "far end opposite the main exterior wall")
  4) estimate_bbox: approximate rounded bounding box (x=[x1,x2], y=[y1,y2], integers), must lie within main polygon's bounding box
  5) estimate_area: estimated area (m², numeric)
  6) layout_reason: rationale for placement (active/quiet zoning, public/private zoning, adjacency, sunlight/daylighting, etc.)
- Evaluate whether the bounding box can roughly accommodate the room and adjust accordingly; if the room has an area in the topo, use it as a reference.
- Output only the round-1 cut; do not cut nodes not in the target set.
        - Follow the bubble diagram relationships when cutting; cuts near elevator/stair must ensure uninterrupted circulation; rooms should be placed as far from stair/elevator as possible without compromising function.
- Later-cut nodes must be checked against previously cut nodes to avoid positional overlap.

Circulation and core clearance constraints (strict):
- Keep distance from cores: all estimate_bbox must NOT intersect stair/elevator rectangles, and must maintain at least 1 grid (≈1m) clearance from their boundaries.
- Preserve main corridor: a continuous, passable corridor (width ≥ 1 grid) must be kept between main and adjacent spaces; no estimate_bbox may block it.
        - Prefer far-side placement: prioritize placing candidate rooms in wings/corners far from the core (stair/elevator) to avoid creating narrow "bottlenecks" or dead ends around the core.
- If a room must be placed near the core, explain the preserved corridor direction and width in layout_reason, and ensure estimate_bbox does not cross the core rectangle's projection lines.

Main node and core adjacency recommendations:
- main should maintain convenient access to stair/elevator; keep at least 1 grid as a buffer zone around the core for main.
- No other room may be placed within this buffer zone around the core; that area belongs to main (corridor/shared space).

Pre-output self-check (must satisfy before outputting):
- Every estimate_bbox lies within main's bounding box;
- No intersection with stair/elevator and clearance is satisfied;
- items do not overlap each other (boundary-only contact is allowed);
- Main circulation path is not blocked.

Area / weight:
- You must provide estimate_area for main and all target nodes; these values will be proportionally normalized to obtain each ratio, with Σ=1 guaranteed;
- Do NOT output rooms or areas for stair/elevator; they are reference-only and do not participate in the cut.


Reference information:
{bbox_hint}
{cores_info}
Floor layout JSON:
```json
{layout_json}
```

Floor topology JSON (nodes and edges):
```json
{topo_json}
```

Round-1 candidate targets (filtered by adjacency to main, excluding elevator/stair):
main node ID: {main_id}
```json
{targets_json}
```

[Output JSON template] Output strictly following the structure and fields below (items MUST include main):
{{
  "round": 1,
  "cut_from": "main",
  "items": [
    {{
      "room_id": "main",
      "estimate_location": "...",
      "estimate_bbox": {{"x": [4,14], "y": [14,19]}},
      "estimate_area": 44.0,
      "layout_reason": "..."
    }}
  ]
}}
"""
