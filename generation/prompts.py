from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .prompts_unused import (
    CUT_PLANNER_TEMPLATE,
    ceiling_selection_prompt,
    floor_plan_prompt,
    object_selection_prompt,
    small_object_selection_prompt,
    wall_height_prompt,
    wall_object_selection_prompt,
)


doorway_prompt = """I need assistance in designing the connections between rooms. The connections could be of three types: doorframe (no door installed), doorway (with a door), or open (no wall separating rooms). The sizes available for doorframes and doorways are single (1m wide) and double (2m wide). When either connected room has a narrow span (width or length under 3 meters), you must avoid double doors and choose single or doorframe instead to preserve usable space.

SPECIAL REQUIREMENTS FOR VERTICAL CORES:
- Stairs (stair): Connections to stair rooms MUST use "doorframe" (no door installed, only doorframe). Do NOT use "doorway" for stairs.
- Elevators (elevator): Connections to elevator rooms MUST use "doorway" (with a door installed) and MUST use "double" size (2m wide). Elevators require double doors for safety, functionality, and accessibility.

Ensure that the door style complements the design of the room. The output format should be: room 1 | room 2 | connection type | size | door style. For example:
exterior | living room | doorway | double | dark brown metal door
living room | kitchen | open | N/A | N/A
living room | bedroom | doorway | single | wooden door with white frames
living room | stair_1 | doorframe | single | N/A


{floor_info}The design under consideration is {input}, which includes these rooms: {rooms}. The length, width and height of each room in meters are:
{room_sizes}
{required_pairs_section}Certain pairs of rooms share a wall: {room_pairs}.{exterior_requirement}
Adhere to these additional requirements: {additional_requirements}.
Provide your response succinctly, without additional text at the beginning or end."""


window_prompt = """Guide me in designing the windows for each room. The window types are: fixed, hung, and slider.
The available sizes (width x height in cm) are:
fixed: (92, 120), (150, 92), (150, 120), (150, 180), (240, 120), (240, 180)
hung: (87, 160), (96, 91), (120, 160), (130, 67), (130, 87), (130, 130)
slider: (91, 92), (120, 61), (120, 91), (120, 120), (150, 92), (150, 120)

Your task is to determine the appropriate type, size, and quantity of windows for each room, bearing in mind the room's design, dimensions, and function.

Please format your suggestions as follows: room | wall direction | window type | size | quantity | window base height (cm from floor). For example:
living room | west | fixed | (130, 130) | 1 | 50

I am now designing {input}. The wall height is {wall_height} cm. The walls available for window installation (direction, width in cm) in each room are:
{walls}
Please note: It is not mandatory to install windows on every available wall. Within the same room, all windows must be the same type and size.
Also, adhere to these additional requirements: {additional_requirements}.

Provide a concise response, omitting any additional text at the beginning or end. """


object_constraints_prompt = """You are an experienced room designer.
Please help me arrange objects in the room by assigning constraints to each object.
Here are the constraints and their definitions:
1. global constraint:
    1) edge: at the edge of the room, close to the wall, most of the objects are placed here.
    2) middle: not close to the edge of the room.

2. distance constraint:
    1) near, object: near to the other object, but with some distance, 50cm < distance < 150cm.
    2) far, object: far away from the other object, distance >= 150cm.
    
3. position constraint:
    1) in front of, object: in front of another object.
    2) around, object: around another object, usually used for chairs.
    3) side of, object: on the side (left or right) of another object.
    4) left of, object: to the left of another object.
    5) right of, object: to the right of another object.
    6) paired, object: strictly paired with another object to form a functional set (e.g., chair-0 paired with desk-0). This constraint forces the object to be placed as close as possible to the target without colliding.

4. alignment constraint:
    1) center aligned, object: align the center of the object with the center of another object.

5. Rotation constraint:
    1) face to, object: face to the center of another object.

6. Matrix constraint:
    1) matrix(rows=N, cols=M, h_gap=X, v_gap=Y): arrange identical objects in a neat, professional grid. 
       - **Usage**: Perfect for e.g. library bookshelves, classroom desks, office workstations, or supermarket aisles.
       - **Parameters**: rows/cols define the grid size; h_gap/v_gap are horizontal/vertical gaps in cm.
       - **Gap sizing with paired items**: When the matrix object has a paired companion (e.g. desk paired with chair), the gap between rows (v_gap) MUST accommodate the paired item plus clearance: v_gap >= paired_item_depth + 60cm. Example: desk with a 60cm-deep chair → v_gap >= 120cm. For face-to-face desk rows sharing a central aisle, v_gap >= 2 × (chair_depth + 30cm).
       - **Gap sizing without paired items**: Bookshelves, cabinets, or display shelves can use small gaps (0–40cm). Passage aisles between columns (h_gap) should be >= 80cm if people walk between them.

For each object, you can assign constraints based on these architectural principles:
1. **Hard Global Constraints**: 
   - `edge`: The object MUST be placed against a wall. Use this for cabinets, beds, or wardrobes. If no wall space is available, the object will be REMOVED.
   - `middle`: The object prefers the center but can fall back to a wall.
   - **Optional**: Not every object needs a global constraint. Items like chairs or coffee tables can rely solely on relative constraints (e.g., `around` or `near`).
2. **Matrix vs Single**: 
   - For objects marked as `placement_type: "matrix"` in the input JSON, you should NOT use suffixes like `-0`. Just use the base name (e.g. `office_desk`).
   - For objects marked as `placement_type: "paired"`, use the base name (e.g. `task_chair`) and do NOT use the `matrix()` constraint. The system will handle the 1-to-1 matching automatically.
   - For objects marked as `placement_type: "single"`, you MUST use indexed names (e.g. `sofa-0`, `bed-0`) even if there is only one.
3. **Paired ONLY for Matrix Groups**: The `paired` constraint is strictly for items belonging to a `matrix` group (e.g. chairs for workstation desks). It ensures that every instance in the matrix has its partner. IMPORTANT: Do not provide `matrix()` parameters for paired items; only provide them for the main anchor object.
4. **Edge Item Restrictions**: Objects with an `edge` global constraint MUST ONLY use `near`, `matrix`, or no other relative constraints. Do NOT use `side of`, `left of`, `right of`, `face to`, etc. for `edge` items. `near` is used to create chains of items that should be placed together along the wall.
5. **Natural Grouping for Single items**: For non-matrix items (e.g. a single bed and its nightstand), do NOT use the `paired` constraint. Use `near` instead.


The output format must be:
object | global constraint (optional) | constraint 1 | constraint 2 | ...
For example:
bookshelf | edge | matrix(rows=1, cols=5, h_gap=5, v_gap=0)
office_desk | middle | matrix(rows=3, cols=5, h_gap=150, v_gap=130)
task_chair | | paired, office_desk | face to, office_desk
bed-0 | edge
nightstand-0 | edge | near, bed-0
sofa-0 | edge
coffee_table-0 | | in front of, sofa-0

Here are some guidelines for you:
1. **Functional Grouping**: Please organize objects into distinct functional groups (e.g., a bed group, a desk group, a seating area) based on their usage.
2. **Matrix Efficiency**: When a room requires many identical items (like bookshelves, desks, or display shelves), the `matrix` constraint ensures perfectly straight alignment and efficient space usage.
3. **Independence**: Each group should start with its own "anchor object" (usually the largest one in the group) that only has a global constraint (edge/middle). Do NOT create a single connected chain for all objects in the room.
4. Place the larger objects first.
4. I will use your guideline to arrange the objects *iteratively*, so the latter objects could only depend on the former objects.
5. The objects of the *same type* are usually *aligned*.
6. I prefer objects to be placed at the edge (the most important constraint) of the room if possible which makes the room look more spacious.
7. **SPECIAL: toilet_suite MUST use `edge` constraint with `matrix(rows=1, cols=N, h_gap=0, v_gap=0)`**. Example: `toilet_suite | edge | matrix(rows=1, cols=3, h_gap=0, v_gap=0)`
8. **IMPORTANT for toilet_suite**: Do NOT add `near, toilet_suite-X` constraints for small items like sanitary_bin, waste_bin, etc. The toilet_suite already includes built-in sanitary bin and accessories. Focus "near" constraints on other anchors like vanity_counter or entrance area.
9. **Decorative/Auxiliary Items**: For standalone decorative items like floor lamps, potted plants, coat racks, floor vases, umbrella stands, or trash cans—placing them in the middle of a room can look awkward and obstruct traffic flow. Using `edge` is usually the safer, more natural choice. However, if there's a clear design intent (e.g., a plant as a room divider, or a floor lamp deliberately placed beside a central seating area), `middle` can work too. When in doubt, prefer `edge`.
10. **CRITICAL - Chairs/Seating NEVER use matrix alone**: Chairs/stools should NEVER be placed in a matrix formation by themselves. A grid of chairs without tables is unnatural and bizarre. Chairs must be `paired` with their corresponding table/desk (e.g., `task_chair | | paired, office_desk`), or placed as `single` items near other furniture (e.g., an armchair near a sofa). The ONLY exception is lecture halls, auditoriums, or waiting areas where rows of audience/waiting chairs are the design intent.

Now I want you to design {room_type} and the room size is {room_size}.
Here are the objects that I want to place in the {room_type}:
{objects}
Please first use natural language to explain your high-level design strategy, and then follow the desired format *strictly* (do not add any additional text at the beginning or end) to provide the constraints for each object."""


wall_object_constraints_prompt = """You are an experienced room designer.
Please help me arrange wall objects in the room by providing their relative position and distance from the floor.
The output format must be: wall object | above, floor object  | distance from floor (cm). For example:
painting-0 | above, sofa-0 | 160
switch-0 | N/A | 120
Note the distance is the distance from the *bottom* of the wall object to the floor. The second column is optional and can be N/A. The object of the same type should be placed at the same height.
Now I am designing {room_type} of which the wall height is {wall_height} cm, and the floor objects in the room are: {floor_objects}.
The wall objects I want to place in the {room_type} are: {wall_objects}.
Please do not add additional text at the beginning or in the end."""


object_selection_prompt_new_1 = """You are an experienced room designer, please assist me in selecting large *floor*/*wall* objects and small objects on top of them to furnish the room. You need to select appropriate objects to satisfy the customer's requirements.
You must provide a description and desired size for each object since I will use it to retrieve object. If multiple items are to be placed in the room with the same description, please indicate the quantity and variance_type ("same" if they should be identical, otherwise "varied").
Present your recommendations in JSON format:
{
    object_name:{
        "description": a short sentence describing the object,
        "location": "floor" or "wall",
        "size": the desired size of the object, in the format of a list of three numbers, [length, width, height] in centimeters,
        "quantity": the number of objects (int). Note: if paired_with is set, this is the quantity PER target object (e.g. 1 chair per desk),
        "variance_type": "same" or "varied",
        "placement_type": "single" (for independent items), "matrix" (for items neatly aligned in rows/cols), or "paired" (strictly for items that belong to a "matrix" group, e.g. chairs for a grid of desks),
        "paired_with": if placement_type is "paired", specify the matrix object_name it belongs to (e.g. "office_desk"), otherwise null,
        "objects_on_top": a list of small children objects (can be empty). See child object format below.
    }
}

**IMPORTANT: objects_on_top child format** (must include ALL fields with proper quotes):
[
    {"object_name": "book", "quantity": 2, "variance_type": "varied"},
    {"object_name": "mug", "quantity": 1, "variance_type": "same"}
]
Note: Each child must have "object_name" (string), "quantity" (int), and "variance_type" ("same" or "varied"). DO NOT forget closing quotes!

Guidelines for placement_type:
1. **Wall Objects**: All objects with `location: "wall"` MUST use `placement_type: "single"`.
2. **Floor Matrix**: Use `matrix` for floor items that need professional row/column grouping (e.g., bookshelves, workstation desks, display shelves).
3. **Paired for Matrix**: Use `paired` ONLY for items that are part of a `matrix` set (e.g. 1 chair for every desk in a matrix). The `quantity` for a paired item represents "how many per matrix member".
4. **Standalone Pairs**: For standalone combinations (e.g. one sofa and one coffee table), do NOT use `paired` placement type. Use `single` for both and link them later via position constraints like `near` or `in front of`.

**CRITICAL - Chairs/Seating Matrix Restrictions**:
- **NEVER use `matrix` for chairs/stools alone** in regular rooms (bedrooms, living rooms, dining rooms, offices, kitchens, etc.). Chairs arranged in a grid without accompanying tables/desks look unnatural and bizarre.
- Chairs should ALMOST ALWAYS be `paired` with their corresponding table/desk, or `single` for standalone seating (like an armchair next to a sofa).
- **Exception - ONLY use `matrix` for chairs** in these SPECIFIC scenarios:
  * Lecture halls / Auditoriums / Theaters (audience seating)
  * Waiting areas / Airport lounges (rows of waiting chairs)
  * Stadium seating / Bleachers
  * Religious venues (pews/prayer seating)
- If you're unsure, default to `paired` (with a table/desk) or `single`. A single chair next to a wall is better than a weird grid of chairs.

For example:
{
    "office_desk": {
        "description": "modern wooden office desk",
        "location": "floor",
        "size": [140, 70, 75],
        "quantity": 6,
        "variance_type": "same",
        "placement_type": "matrix",
        "paired_with": null,
        "objects_on_top": [
            {"object_name": "laptop", "quantity": 1, "variance_type": "same"},
            {"object_name": "desk lamp", "quantity": 1, "variance_type": "same"},
            {"object_name": "pen holder", "quantity": 1, "variance_type": "varied"}
        ]
    },
    "task_chair": {
        "description": "ergonomic office chair",
        "location": "floor",
        "size": [60, 60, 100],
        "quantity": 1,
        "variance_type": "same",
        "placement_type": "paired",
        "paired_with": "office_desk",
        "objects_on_top": []
    },
    "painting": {
        "description": "abstract wall painting",
        "location": "wall",
        "size": [80, 2, 60],
        "quantity": 2,
        "variance_type": "varied",
        "placement_type": "single",
        "paired_with": null,
        "objects_on_top": []
    },
    "wall_clock": {
        "description": "minimalist round wall clock",
        "location": "wall",
        "size": [30, 5, 30],
        "quantity": 1,
        "variance_type": "same",
        "placement_type": "single",
        "paired_with": null,
        "objects_on_top": []
    },
    "sofa": {
        "description": "modern sectional, light grey sofa",
        "location": "floor",
        "size": [200, 100, 80],
        "quantity": 1,
        "variance_type": "same",
        "placement_type": "single",
        "paired_with": null,
        "objects_on_top": [
            {"object_name": "throw pillow", "quantity": 3, "variance_type": "varied"},
            {"object_name": "blanket", "quantity": 1, "variance_type": "same"}
        ]
    },
    "coffee_table": {
        "description": "modern glass coffee table",
        "location": "floor",
        "size": [120, 60, 45],
        "quantity": 1,
        "variance_type": "same",
        "placement_type": "single",
        "paired_with": null,
        "objects_on_top": [
            {"object_name": "magazine", "quantity": 3, "variance_type": "varied"},
            {"object_name": "remote control", "quantity": 2, "variance_type": "same"},
            {"object_name": "decorative vase", "quantity": 1, "variance_type": "same"}
        ]
    }
}

You are now designing the room "*CURRENT_ROOM_ID*" for the project "*INPUT*". 
Other rooms on this floor include: [*OTHER_ROOMS*].
Please also consider the following additional requirements: REQUIREMENTS.

Here are some guidelines for you:
1. Provide reasonable type/style/quantity of objects for each room based on the room size to make the room not too crowded or empty.
2. Do not provide rug/mat, windows, doors, curtains, and ceiling objects which have been installed for each room.
3. Large furniture should share a coherent style/material/color palette within the room; small accessories can be diverse/varied for liveliness.
4. Maintain style consistency: for the same category (e.g., chairs/seating), keep styles similar and avoid mixing too many different chair types in one room; neutral variety like different books is acceptable.
5. Seats (chairs/stools/sofas) must be separate floor objects, NOT in objects_on_top. Child items in objects_on_top must be small tabletop/shelf items only (books, mugs, lamps, decorations, electronics like laptops/phones, etc.).
6. Use objects_on_top to add life and detail to the room - tables should have items on them, shelves should have decorations, desks should have office supplies.

**SPECIAL RULE for Public Restrooms/Toilets**:
For restrooms, toilets, or bathrooms in PUBLIC buildings (offices, schools, hospitals, shopping malls, airports, train stations, libraries, museums, gyms, stadiums, hotels, etc.), you MUST follow these rules:
- **IMPORTANT**: The object name MUST be EXACTLY "toilet_suite" (no variations like "toilet_suite_standard", "toilet_stall", etc.). This exact name is required for our asset retrieval system.
- **toilet_suite is a COMPLETE UNIT** that already includes: partition walls, toilet bowl, toilet paper holder, coat hook, and sanitary bin. You do NOT need to add any items near or inside the toilet stall zone (no extra sanitary_bin, waste_bin, toilet_paper_holder, etc. near toilet_suite).
- **Focus on OTHER areas** of the restroom instead: vanity/sink counter area, entrance/lobby area, storage lockers, hand dryers, mirrors, etc.
- Set `placement_type: "matrix"` with `quantity` based on room size.
- The toilet suite MUST be arranged in a single row against a wall with ZERO gap: matrix(rows=1, cols=N, h_gap=0, v_gap=0). Do NOT use multiple rows. h_gap MUST be 0 because the partitions are already built into the asset.
- COPY THIS EXAMPLE EXACTLY for public restrooms (only change quantity):
  ```json
  "toilet_suite": {
      "description": "toilet suite",
      "location": "floor",
      "size": [166, 200, 141],
      "quantity": 3,
      "variance_type": "same",
      "placement_type": "matrix",
      "paired_with": null
  }
  ```
- For residential/private bathrooms (single-family homes, apartments, hotel guest rooms), use regular "toilet" instead.

Please first use natural language to explain your high-level design strategy for the room type *CURRENT_ROOM_ID*, and then provide your recommendations in a single JSON code block (using ```json ... ``` strictly).

**CRITICAL JSON FORMATTING**: Ensure all string values have proper opening AND closing quotes (e.g., "varied" not "varied). Double-check your JSON is valid before submitting."""


object_selection_prompt_new_2 = """User: {object_selection_prompt_new_1}

Agent: {object_selection_1}

User: Thanks! After following your suggestions to retrieve objects, I found the *{room}* is still too empty. To enrich the *{room}*, you could:
1. Add more *floor* objects to the *{room}* (excluding rug, carpet, windows, doors, curtains, and *ignore ceiling objects*).
2. Increase the size and quantity of the objects.
Could you update the entire JSON file with the same format as before and answer without additional text at the beginning or end?

Agent: """


floor_baseline_prompt = """
You are an experienced room designer.

You operate in a 2D Space. You work in a X,Y coordinate system. (0, 0) denotes the bottom-left corner of the room.
All objects should be placed in the positive quadrant. That is, coordinates of objects should be positive integer in centimeters.
Objects by default face +Y axis.
You answer by only generating JSON files that contain a list of the following information for each object:

- object_name: name of the object, follow the name strictly.
- position: coordinate of the object (center of the object bounding box) in the form of a dictionary, e.g. {{"X": 120, "Y": 200}}.
- rotation: the object rotation angle in clockwise direction when viewed by an observer looking along the z-axis towards the origin, e.g. 90. The default rotation is 0 which is +Y axis.

For example: {{"object_name": "sofa-0", "position": {{"X": 120, "Y": 200}}, "rotation": 90}}

Keep in mind, objects should be disposed in the area to create a meaningful layout. It is important that all objects provided are placed in the room.
Also keep in mind that the objects should be disposed all over the area in respect to the origin point of the area, and you can use negative values as well to display items correctly, since origin of the area is always at the center of the area.

Now I want you to design {room_type} and the room size is {room_size}.
Here are the objects (with their sizes) that I want to place in the {room_type}:
{objects}

Remember, you only generate JSON code, nothing else. It's very important. Respond in markdown (```).
"""

# --- Object Refiner ---
object_refine_prompt = """You are a senior interior space auditor. I have a preliminary furniture plan for a room, along with potential 3D asset options retrieved for each item.

TASK:
Audit each item in the JSON based on **SEMANTIC MATCHING ONLY**:

1. **Semantic Selection**: Compare the 'retrieved_asset_options' with your 'description' (design intent). 
   - **Pick the best matching asset** and set `chosen_option_index` to its index.
   - Update the `size` field to match the `actual_physical_size_cm` of the chosen option.
   - **DO NOT change the `quantity`** - keep it as specified in the original plan.

2. **Discard (ONLY when semantically unacceptable)**: 
   - If NO options match the intent AT ALL (e.g., you wanted a 'toilet' but only got 'plunger'), remove the item.
   - **Be lenient**: If an option is "close enough" or "acceptable substitute", KEEP it.
   - Example: a 'painting' for 'mirror' is NOT acceptable (different function), but 'kitchen sink' for 'vanity with sink' is acceptable.

3. **Objects On Top**: 
   - Review the `objects_on_top` list for each item.
   - Remove ONLY items that are NOT suitable small tabletop/shelf objects (e.g., remove chairs, stools, large appliances).
   - Keep quantities as specified - do NOT adjust based on parent size.

INPUT DATA:
{plan_with_assets}

OUTPUT FORMAT (JSON ONLY):
Return the updated JSON structure. For each item, you MUST include `chosen_option_index` and `objects_on_top` (can be empty list).
Keep `quantity` unchanged from the input.
Example:
{{
  "office_desk": {{
    "description": "modern wooden desk",
    "location": "floor",
    "size": [140, 70, 75],
    "quantity": 4,
    "chosen_option_index": 1,
    "objects_on_top": [
      {{"object_name": "laptop", "quantity": 1, "variance_type": "same"}},
      {{"object_name": "desk lamp", "quantity": 1, "variance_type": "same"}}
    ]
  }},
  "sofa": {{
    "description": "modern grey sofa",
    "location": "floor",
    "size": [200, 90, 80],
    "quantity": 1,
    "chosen_option_index": 0,
    "objects_on_top": [
      {{"object_name": "throw pillow", "quantity": 2, "variance_type": "varied"}}
    ]
  }}
}}
"""

# --- Object Refiner Fallback ---
object_refine_fallback_prompt = """You are a senior interior designer helping to find alternatives for furniture items that failed retrieval.

CONTEXT:
- Room type: {room_type}
- Some items could not be found in the 3D asset database
- We need to either suggest better search terms OR suggest replacement items

FAILED ITEMS:
{failed_items_json}

YOUR TASK:
For each failed item, provide ONE of these solutions:
1. **alternative_search_terms**: List 2-4 alternative English search terms that might find the intended object
   - Use more generic terms (e.g., "toilet" -> "bathroom fixture", "WC", "commode")
   - Use different phrasing (e.g., "wall mirror" -> "rectangular mirror", "vanity mirror")
   
2. **replacement_item**: If the item is too specific, suggest a functionally similar item that is more likely to exist
   - Keep the same location (floor/wall)
   - Maintain similar function in the room
   - Use common furniture names

IMPORTANT RULES:
- Prefer alternative_search_terms over replacement_item (try to keep original design intent)
- Only use replacement_item if the original item is truly obscure or specialized
- Search terms should be in English
- Be practical - suggest items commonly found in 3D asset libraries

OUTPUT FORMAT (JSON ONLY):
{{
  "toilet": {{
    "solution_type": "alternative_search_terms",
    "alternative_search_terms": ["bathroom toilet", "WC fixture", "ceramic toilet", "lavatory"],
    "reason": "Original term may be too generic, trying more specific bathroom fixture terms"
  }},
  "custom_art_piece": {{
    "solution_type": "replacement_item",
    "replacement_item": {{
      "object_name": "wall_painting",
      "description": "framed abstract painting for wall decoration",
      "location": "wall",
      "size": [80, 4, 60]
    }},
    "reason": "Custom art is too specific, replacing with common wall art"
  }}
}}
"""

# --- Topology Planner ---
TOPOLOGY_PLANNER_TEMPLATE = """You are an experienced architect designing a single floor's topological connectivity. Each node must include an area estimate (m²). Do NOT invent an id "main"; instead, choose one existing node to act as the traffic hub by setting its type to "main" (e.g., living/atrium/lobby). If a space is clearly dominant in area, promote it as main even if the hints suggest otherwise.

Inputs
Overall program reasoning:
{reasoning}
{user_requirement_text}
Floor context
- Floor index: {idx}
- Gross floor area (reference): {gfa}
- Floor requirement: {requirement}
- Floor polygon JSON (main space after cores removed):
```json
{layout_json}
```

 {vtext}

Material selection guidance
{material_hints_text}
- Material selection principles:
  * Prioritize materials specified in the building_program for each room (see material reference above)
  * If no material is specified in the building_program, choose appropriate materials based on room function and building type
  * Rooms of the same type should maintain consistent materials
  * Stairs and elevators should share the same materials as the main space
  * Wet areas such as restrooms and tea rooms must use waterproof, non-slip materials (e.g., non-slip ceramic tile)
  * Every node must include floor_material and wall_material fields using descriptive text (e.g., 'warm oak hardwood, matte', 'soft beige drywall, smooth', 'non-slip ceramic tile', etc.)

Floor hints from program:
{rooms_json}

Your task
- Include vertical connectors present in the layout (elevator/stair) as fixed nodes for this floor. Do NOT omit elevator/stair on any floor.
- Derive a minimal useful set of rooms/spaces for this floor based on the requirement and rooms list below; capture only abstract connectivity (not geometry). Treat the provided rooms list as hints (first item is the suggested circulation hub), but you should re-evaluate who is the true "main" if one space's area clearly dominates the floor.

Hard requirements
- MANDATORY: If the floor layout shows any elevator or stair cores, you MUST include corresponding topo nodes (type="elevator" / type="stair") in this floor's topology and label them with open_relation="door". Do not drop elevator/stair on upper floors. Keep their footprint consistent with the layout; if uncertain, keep a 2m x 2m (area=4) core at the hinted bbox.
- Node types allowed: main, Entities, area, elevator, stair.
- Semantic hierarchy constraint: Entities can only act as "leaf nodes"; do not attach type="area" functional-zone nodes beneath an Entities node. If further grouping is needed, use an area node at a higher level instead.
- area can have several Entities attached, and may also attach other area nodes as sub-functional-zones when necessary; however, Entities should NOT act as the "parent" of an area.
- Edge kinds allowed: access (for connectors like main↔stair/elevator), adjacent (general adjacency). In topological terms, main↔area↔Entities can be understood as main being directly adjacent to those Entities.
- Output strictly JSON with nodes (id, type, area, floor_material, wall_material, open_relation) and edges.
- open_relation field: every node must include an open_relation field with value either "open" or "door". For type="area" nodes, open_relation has little impact on final geometry and can simply be set to "door".
  * For the main node, it MUST be set to "open"
  * For elevator and stair nodes, it MUST be set to "door"
  * For other rooms (Entities, area), decide based on room function and privacy requirements:
    - "open": open space, no door (e.g., open-plan office, lobby, etc.)
    - "door": enclosed space requiring a door (e.g., bedroom, meeting room, restroom, etc.)
  * Please choose an appropriate open_relation value for each room; the system will only enforce corrections for main, elevator, and stair — all other rooms follow your output exactly.
- Area constraints:
  * Using an area node as a full partition wrapper means its children fully partition it: sum(child.area) == area.area
  * area is only for equal/clear partitioning of a functional zone. If children are partial or overlapping concepts, do not use area as a full partition wrapper.
  * adjacent indicates carving area from its parent context (typically main). Keep a reasonable main (circulation) reserve.

Planning guidance
- A single node (including main, Entities, area) should generally have no more than 6 direct child connections; when there are many semantically adjacent or functionally similar rooms, prefer grouping them through an intermediate functional-zone node rather than hanging a dozen rooms directly under the same node at once.
- For multiple rooms of similar nature and area arranged side by side (e.g., 3–8 equally-sized meeting rooms, examination rooms, wards), prefer using a single node of type "area" to represent this functional zone, with specific Entities rooms attached beneath it; subsequent geometry cuts will fully partition that area into those entity rooms, and the area node itself will not appear in the final floor plan nor retain any leftover area.
- Guidelines for using area vs. Entities (must follow):
  * When child rooms all need to be directly adjacent to the parent node (side by side, sharing a hall/corridor, same functional zone), use area to wrap those rooms; children of an area mostly need to connect directly to their parent. For example: {{"area": [classroom_1, classroom_2]}}, {{"area": [kitchen, dining_area]}}.
  * When child rooms are a secondary subdivision within the parent, mainly needing adjacency only to the parent without needing to connect back up to higher-level nodes, use Entities chain decomposition rather than area. For example, bedroom→bathroom should be written as bedroom (Entities) with bathroom (also Entities) attached beneath it, rather than placing bedroom + bathroom side by side inside an area.
  * When in doubt, prefer Entities (the more conservative "chain" topology) to avoid overusing area, which creates too many "must be directly adjacent to parent" constraints.

Output schema
{{
  "nodes": [
    {{"id": "lobby", "type": "main", "area": 80.0, "floor_material": "warm oak hardwood, matte", "wall_material": "soft beige drywall, smooth", "open_relation": "open"}},
    {{"id": "office_zone", "type": "area", "area": 30.0, "floor_material": "warm oak hardwood, matte", "wall_material": "soft beige drywall, smooth", "open_relation": "door"}},
    {{"id": "room_1", "type": "Entities", "area": 15.0, "floor_material": "carpet, neutral gray", "wall_material": "painted drywall, white", "open_relation": "door"}},
    {{"id": "room_2", "type": "Entities", "area": 15.0, "floor_material": "non-slip ceramic tile", "wall_material": "waterproof tile, white", "open_relation": "door"}}
  ],
  "edges": [{{"source": "lobby", "target": "office_zone", "kind": "adjacent"}}, {{"source": "office_zone", "target": "room_1", "kind": "adjacent"}}]
}}
"""

# --- Building Program ---
BUILDING_PROGRAM_TEMPLATE = """You are an experienced architect asked to program a complete multi-floor building using the first-floor plan as reference.

Inputs
- First-floor boundary and area (JSON below) and an image reference
- Number of floors: {floors}
- {user_part}
{area_note}

Key rule
- {base_rule}
{height_rule}
{material_blocks}

Core placement preferences
- Place stairs/elevators only in corners that are bounded by two exterior walls, and pick the corner whose surrounding leftover space is smallest, so large continuous areas stay intact.
- Make use of tight/awkward leftover pockets that are hard to program otherwise.
- Quantize core boxes to integer coordinates and size exactly 2x2. Coordinates must be non-negative.

Room programming guidelines
- Keep room types practical and necessary. Avoid adding small auxiliary rooms (like walk-in closets, pantries, mudrooms) unless explicitly requested by the user or clearly justified by building function.
- For residential bedrooms, storage can be handled by wardrobes/closets as furniture; a separate walk_in_closet room is NOT needed unless the user specifically asks for it.
- Prioritize larger, multi-purpose spaces over many small fragmented rooms.

What to do
1) Analyze the outline and area and reason about a practical building function.
2) Decide the vertical connectivity method: stair | elevator | stair_and_elevator, consistent with the rule.
3) Choose locations for vertical cores; each stair/elevator occupies an axis-aligned 2x2 bbox within the floor polygon, ideally on an exterior edge/corner. Output as x=[x1,x2], y=[y1,y2] with integer coordinates.
{material_step}
5) Room sums must not exceed the gross floor area; keep 12–25% as circulation/core reserve unless justified.
6) Ensure totals are reasonable; indicate whether plans fit within GFA after reserving circulation.
7) If a floor layout should be exactly the same as another floor, use a shorthand: specify only {{"index": k, "copy": j}} meaning “floor k copies floor j”. When using copy, do not include other fields for that floor.

Output strictly JSON only (no Markdown fences) with this exact schema:
{{
  "reasoning": "brief explanation of program and area logic",
  "vertical_connectivity": {{
    "method": "stair | elevator | stair_and_elevator",
    "cores": [
      {{"type": "stair", "x": [7,9], "y": [2,4]}}
    ],
    "justification": "why this choice fits the rule and requirement"
  }},
  "floors": [
    {{
      "index": 1,
      "requirement": "natural language requirement for floor 1",
      "gross_floor_area": 0.0,
      "rooms": [
{room_schema}
      ],
      "area_summary": {{
        "sum_rooms": 0.0,
        "reserve_ratio": 0.18,
        "fits_within_gfa": true,
        "notes": ""
      }}
    }}
  ]
}}

First-floor JSON:
```json
{layout_json}
```
"""

# --- Material Planner ---
MATERIAL_PLANNER_TEMPLATE = """You are an interior material strategist.

Goal: Based on the building program below (no materials yet), assign floor_material and wall_material for each room type. Keep the palette small and consistent across the entire building so same room types reuse the same materials across floors. Wet areas (bathroom, restroom, shafts) must use moisture-resistant, non-slip finishes. Stair/elevator cores should follow the public/common area material.

Constraints:
- Output no more than 5 distinct floor materials and 5 wall materials; reuse whenever possible.
- Do NOT add or remove rooms, floors, or other fields. You only decide materials.
- If a floor uses {{"copy": k}}, assume it copies materials from that referenced floor; do not expand it.
- Materials should be short descriptive strings (e.g., "matte light oak wood", "smooth warm white paint", "non-slip gray ceramic tile").

Return ONLY JSON with this schema:
{{
  "default_floor_material": "string (fallback for unspecified rooms)",
  "default_wall_material": "string (fallback)",
  "room_materials": {{
    "room_id": {{"floor_material": "...", "wall_material": "..."}}
  }}
}}

Program summary (read-only, do not rewrite):
```json
{summary_json}
```"""


# --- Seed Guidance (LLM growth) ---
SEED_GUIDANCE_TEMPLATE = """You are a floor plan seed planning assistant. Given the floor topology and a preview image containing the full floor outline with stairs/elevators, your task is to plan an axis-aligned rectangular bounding box for each target room within the current parent room, expressed as x=[xmin,xmax], y=[ymin,ymax] to indicate each room's approximate position and extent.
Coordinate convention: x increases to the right, y increases upward; the parent room's approximate coordinate range is x∈[{minx:.1f},{maxx:.1f}], y∈[{miny:.1f},{maxy:.1f}]; your output should use values within this coordinate interval.
The output must be a JSON array where each element contains:
  - room_id: string, the room ID (must be chosen from the candidate list only)
  - x: array of length 2 [xmin,xmax], the bounding box in the x direction, requires xmin <= xmax
  - y: array of length 2 [ymin,ymax], the bounding box in the y direction, requires ymin <= ymax
  - area: float, the approximate fraction of the parent room's area this room occupies (0.0–1.0, optional, used as a downstream hint, does not need to be very precise)
  - reason: a brief one-sentence explanation of why you placed this room at this position.

For example:
[
  {{"room_id": "lobby", "x": [0.0, 8.0], "y": [0.0, 6.0], "area": 0.45, "reason": "Main lobby occupies the central area, enclosing the stair core for direct access."}},
  {{"room_id": "office_1", "x": [8.0, 12.0], "y": [0.0, 5.0], "area": 0.2, "reason": "Office placed at the east wing, away from the stair core."}}
]

{special_instruction}

[Target room list]
{target_ids_text}

[Topology and layout context]
- Requirement: {requirement}
- Parent room ID: {parent_id}
- Parent room type: {parent_type}
- Parent room area (estimated): {parent_area:.1f}
- Adjacent room IDs (already placed): {neighbor_ids_text}

Please refer to the floor outline and existing stair/elevator positions in the preview image to provide a reasonable bounding box allocation.
Note: bounding boxes should be placed as much as possible within the parent room outline; avoid large-area overlaps.
"""
# --- Small Object Refiner ---
small_object_refine_prompt = """You are a senior interior detail auditor. I have a list of small objects (tabletop items) planned for placement on furniture receptacles, along with their retrieved 3D asset candidates.

CONTEXT:
- Room: {room_id}
- These small objects are planned to be placed ON TOP OF furniture that has already been successfully placed in the room.
- Your task is to audit each small object and decide whether to KEEP or DISCARD it based on semantic matching.

TASK:
For each small object, audit based on **SEMANTIC MATCHING ONLY**:

1. **Semantic Selection**: Compare the 'retrieved_asset_options' with the intended 'object_name'.
   - **Pick the best matching asset** and set `chosen_option_index` to its index.
   - If multiple options are acceptable, prefer the one closest to the original intent.

2. **Discard (ONLY when semantically unacceptable)**:
   - If NO options match the intent AT ALL (e.g., you wanted a 'laptop' but only got 'banana'), set `chosen_option_index` to -1 to discard.
   - **Be lenient**: If an option is "close enough" or "acceptable substitute", KEEP it.
   - Example: a 'notebook' for 'book' is acceptable; a 'plant' for 'pen holder' is NOT acceptable.

3. **No Fallback**: If discarded, the item is simply removed. There is no recovery mechanism.

INPUT DATA:
{small_objects_with_assets}

OUTPUT FORMAT (JSON ONLY):
Return a JSON object where each key is the small object identifier, and the value contains `chosen_option_index` (-1 to discard, otherwise the index of the best option).

Example:
{{
  "laptop|desk-0(office_1)": {{
    "chosen_option_index": 0,
    "reason": "First option is a laptop, matches intent perfectly"
  }},
  "pen_holder|desk-0(office_1)": {{
    "chosen_option_index": 2,
    "reason": "Third option is a pencil cup, acceptable substitute for pen holder"
  }},
  "exotic_plant|shelf-0(office_1)": {{
    "chosen_option_index": -1,
    "reason": "No options match a plant at all, discarding"
  }}
}}
"""

# --- Core Repair ---
# --- Core Repair ---
CORE_REPAIR_TEMPLATE = """You are a floor plan correction assistant. For the given invalid elevator/stair cores only, return corrected axis-aligned positions that satisfy:
- Maintain {size}x{size} size, integer coordinates, fully within the main polygon
- No mutual overlap
- Only return cores that need modification (id and type are required); do not output any other fields

[main polygon]
{main_polygon_json}

[Current core list (reference only)]
{current_cores_json}

[Invalid cores and issues]
{invalids_json}

[Output JSON format]
{{
  "cores": [ {{"id": "stair_1", "type": "stair", "x": [7,{x_plus_size}], "y": [2,{y_plus_size}]}} ]
}}
Output JSON only, no explanation."""
