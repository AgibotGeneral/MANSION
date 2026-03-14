from procthor.utils.types import RGB, Light, LightShadow, Vector3
from shapely import Polygon


def generate_lights(scene, profile=None):
    lights = [
        Light(
            id="DirectionalLight",
            position=Vector3(x=0.84, y=0.1855, z=-1.09),
            rotation=Vector3(x=43.375, y=-3.902, z=-63.618),
            shadow=LightShadow(
                type="Soft",
                strength=1,
                normalBias=0,
                bias=0,
                nearPlane=0.2,
                resolution="FromQualitySettings",
            ),
            type="directional",
            intensity=0.35,
            indirectMultiplier=1.0,
            rgb=RGB(r=1.0, g=1.0, b=1.0),
        )
    ]

    # Allow global configuration overrides
    profile = profile or {}
    dir_override = profile.get("directional", {})
    point_override = profile.get("point", {})
    if dir_override:
        lights[0]["intensity"] = dir_override.get("intensity", lights[0]["intensity"])
        lights[0]["rgb"] = dir_override.get("rgb", lights[0]["rgb"])
        lights[0]["rotation"] = dir_override.get("rotation", lights[0]["rotation"])

    for room in scene["rooms"]:
        room_id = room["id"]
        floor_polygon = Polygon(room["vertices"])
        x = floor_polygon.centroid.x
        z = floor_polygon.centroid.y

        light_height = scene["wall_height"] - 0.2
        try:
            for object in scene["ceiling_objects"]:
                if object["roomId"] == room_id:
                    light_height = object["position"]["y"] - 0.2
        except:
            light_height = scene["wall_height"] - 0.2

        lights.append(
            Light(
                id=f"light|{room_id}",
                type="point",
                position=Vector3(x=x, y=light_height, z=z),
                intensity=point_override.get("intensity", 0.75),
                range=point_override.get("range", 15),
                rgb=point_override.get("rgb", RGB(r=1.0, g=0.855, b=0.722)),
                shadow=LightShadow(
                    type="Soft",
                    strength=1,
                    normalBias=0,
                    bias=0.05,
                    nearPlane=0.2,
                    resolution="FromQualitySettings",
                ),
                roomId=room_id,
            )
        )

    return lights
