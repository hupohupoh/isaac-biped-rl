"""
Import URDF → USD using Isaac Sim's Python API.

Run this INSIDE Isaac Sim:
    1. Open Isaac Sim
    2. Window → Script Editor
    3. Copy-paste this script and run

Or from command line:
    isaacsim -p F:\RobotProject\biped_demo\import_urdf.py
"""

import os

# ---- CONFIG ----
URDF_PATH = r"F:\RobotProject\biped_demo\biped_clean.urdf"
OUTPUT_USD = r"F:\RobotProject\biped_demo\source\biped_demo\biped_demo\tasks\manager_based\biped_demo\assets\robot\usd\biped.usd"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_USD), exist_ok=True)

print("=" * 60)
print("URDF → USD Converter for Custom Biped Robot")
print("=" * 60)
print(f"URDF:  {URDF_PATH}")
print(f"USD:   {OUTPUT_USD}")
print()

# ---- Method 1: UrdfConverter API ----
try:
    from omni.importer.urdf import UrdfConverter

    print("[1/4] Creating USD stage...")
    import omni.usd
    # Get or create a new stage
    stage = omni.usd.get_context().open_stage(OUTPUT_USD)
    if not stage:
        stage = omni.usd.get_context().create_new_stage(OUTPUT_USD)
    print(f"      Stage: {OUTPUT_USD}")

    print("[2/4] Importing URDF...")
    converter = UrdfConverter()
    # Import URDF with settings matching the GUI recommendations
    success = converter.import_urdf(
        urdf_path=URDF_PATH,
        import_config={
            "merge_fixed_joints": False,
            "import_inertia_tensor": True,
            "convex_decomp": False,
            "self_collision": True,
            "fix_base": False,
            "make_default_prim": True,
            "create_physics_scene": True,
        },
    )

    if success:
        print("[3/4] Saving USD stage...")
        omni.usd.get_context().save()
        print("[4/4] DONE!")
        print(f"\nUSD file saved to: {OUTPUT_USD}")
    else:
        print("ERROR: URDF import failed via UrdfConverter.")
        raise RuntimeError("UrdfConverter.import_urdf() returned False")

except ImportError as e:
    print(f"UrdfConverter not available: {e}")
    print("Trying alternative method...")

    # ---- Method 2: omni.kit.commands ----
    try:
        import omni.kit.commands
        from pxr import Usd, Sdf

        print("[1/3] Creating empty USD stage...")
        import omni.usd
        stage = omni.usd.get_context().create_new_stage(OUTPUT_USD)

        print("[2/3] Importing URDF via kit command...")
        omni.kit.commands.execute(
            "CreateAsset",
            asset_path=URDF_PATH,
            imported_asset_path="/World/biped_robot",
        )

        print("[3/3] Saving...")
        omni.usd.get_context().save()
        print(f"\nUSD file saved to: {OUTPUT_USD}")

    except Exception as e2:
        print(f"Both methods failed.")
        print(f"Method 1 error: {e}")
        print(f"Method 2 error: {e2}")

        # ---- Method 3: Minimal fallback using low-level import ----
        print("\nTrying low-level URDF import...")
        try:
            from isaacsim.robot.urdf import import_urdf as isaac_import_urdf

            isaac_import_urdf(
                urdf_path=URDF_PATH,
                output_path=OUTPUT_USD,
                fix_base=False,
                make_default_prim=True,
            )
            print(f"USD saved to: {OUTPUT_USD}")
        except Exception as e3:
            print(f"All three methods failed: {e3}")
            print("\nPlease try the GUI method:")
            print("  File → Import → URDF → Direct Import")
            print(f"  Select: {URDF_PATH}")
