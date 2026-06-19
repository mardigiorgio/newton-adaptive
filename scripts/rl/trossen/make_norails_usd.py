"""Generate a no-rails override of the Stationary AI rig USD.

The rig's ``frame_link`` is a single body whose collision mesh wraps the whole platform
perimeter (the rails) and rises into the camera gantry. A lift policy learns to jam the cube
against that rail border instead of grasping cleanly. This writes a thin ASCII override layer
that sublayers the original USD and (a) deactivates ``frame_link``'s collision and (b) hides its
visual -- leaving the arms and tabletop untouched. The original asset is never modified.

    uv run --with usd-core python scripts/rl/trossen/make_norails_usd.py

Output sits next to the original so the relative sublayer reference resolves. Point the env at it
(see cube_lift_env_cfg.py: robot spawn usd_path -> stationary_ai_norails.usda).
"""

import os

from pxr import Usd, UsdGeom

ASSET_DIR = os.path.expanduser("~/isaac-rl/trossen_ai_isaac/assets/robots/stationary_ai")
ORIG = os.path.join(ASSET_DIR, "stationary_ai.usd")
OUT = os.path.join(ASSET_DIR, "stationary_ai_norails.usda")


def main() -> None:
    assert os.path.exists(ORIG), f"original rig USD not found: {ORIG}"
    if os.path.exists(OUT):
        os.remove(OUT)

    stage = Usd.Stage.CreateNew(OUT)
    # Sublayer the original (relative path, same dir) so all prim paths compose unchanged;
    # opinions authored here (root layer) are stronger and win.
    stage.GetRootLayer().subLayerPaths.append("./stationary_ai.usd")

    # Rails off: deactivating the collisions scope prunes its meshes from the composed stage.
    stage.OverridePrim("/stationary_ai/frame_link/collisions").SetActive(False)
    # Rails hidden: the body stays in the articulation, just not drawn.
    UsdGeom.Imageable(stage.OverridePrim("/stationary_ai/frame_link/visuals")).MakeInvisible()

    sa = stage.GetPrimAtPath("/stationary_ai")
    assert sa and sa.IsValid(), "sublayer did not compose /stationary_ai"
    stage.SetDefaultPrim(sa)
    stage.GetRootLayer().Save()
    print(f"wrote {OUT}")
    print("frame_link/collisions active:", stage.GetPrimAtPath("/stationary_ai/frame_link/collisions").IsActive())
    print("frame_link/visuals visible:",
          UsdGeom.Imageable(stage.GetPrimAtPath("/stationary_ai/frame_link/visuals")).ComputeVisibility())


if __name__ == "__main__":
    main()
