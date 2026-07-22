"""Compute and visualize the overall center of mass of the biped robot USD.

Usage:
    # In Isaac Sim's Script Editor or Python window:
    python scripts/check_com.py

    # Or run standalone with USD library:
    python scripts/check_com.py --usd biped_clean/biped_clean.usda

This script reads all physics:centerOfMass and physics:mass values from the USD
and computes the weighted overall center of gravity.
"""

import argparse
from pathlib import Path


def compute_overall_com(usd_path: str, joint_positions: dict | None = None):
    """Compute the overall center of mass from a USD file.

    Args:
        usd_path: Path to the robot USD file.
        joint_positions: Optional dict of {joint_name: angle_rad} to set joint angles.
                         If None, uses the default (standing) pose.

    Returns:
        dict with keys: total_mass, overall_com, per_link_coms
    """
    from pxr import Usd, UsdPhysics, Gf, PhysxSchema

    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()

    if not prim:
        raise ValueError(f"No default prim found in {usd_path}")

    # Optionally set joint positions (requires articulation API)
    if joint_positions:
        _set_joint_angles(stage, joint_positions)

    # Collect all rigid bodies with mass properties
    link_data = []
    for rb_prim in stage.Traverse():
        mass_api = UsdPhysics.MassAPI.Apply(rb_prim)
        # Check if this prim has mass (not all prims do)
        if not mass_api or not mass_api.GetMassAttr().Get():
            continue

        mass = mass_api.GetMassAttr().Get()
        com_attr = mass_api.GetCenterOfMassAttr()
        if not com_attr or not com_attr.Get():
            continue

        com_local = Gf.Vec3f(com_attr.Get())
        link_name = str(rb_prim.GetPath())

        # Transform CoM to world frame
        # Navigate up the parent chain to accumulate transforms
        com_world = _local_to_world(rb_prim, com_local)

        link_data.append({
            "name": link_name,
            "mass": mass,
            "com_local": com_local,
            "com_world": com_world,
        })

    # Compute overall CoM
    total_mass = sum(d["mass"] for d in link_data)
    if total_mass == 0:
        return {"total_mass": 0.0, "overall_com": (0, 0, 0), "per_link_coms": link_data}

    overall_com = Gf.Vec3f(0, 0, 0)
    for d in link_data:
        overall_com += d["mass"] * d["com_world"]
    overall_com /= total_mass

    return {
        "total_mass": total_mass,
        "overall_com": (overall_com[0], overall_com[1], overall_com[2]),
        "per_link_coms": link_data,
    }


def _local_to_world(prim, local_point: tuple) -> tuple:
    """Transform a point from a prim's local frame to world frame.

    Walks up the parent chain accumulating xformOp:translate and xformOp:orient.
    """
    from pxr import Gf

    point = Gf.Vec3d(*local_point)
    current = prim

    while current:
        # Check for xformOps
        xform = _get_xform(current)
        if xform:
            translate = xform.get("translate")
            orient = xform.get("orient")
            if orient:
                rot = Gf.Rotation(Gf.Quatd(*orient))
                point = rot.TransformDir(point)
            if translate:
                point += Gf.Vec3d(*translate)
        current = current.GetParent()

    return Gf.Vec3f(point)


def _get_xform(prim) -> dict | None:
    """Extract translate/orient from a prim's xformOps, if any."""
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None

    result = {}
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        name = op.GetOpName()
        if "translate" in name:
            result["translate"] = op.Get()
        elif "orient" in name:
            result["orient"] = op.Get()
    return result if result else None


def _set_joint_angles(stage, joint_positions: dict):
    """Set joint angles on the articulation."""
    from pxr import UsdPhysics

    for joint_name, angle in joint_positions.items():
        joint_path = f"/biped_robot/Physics/{joint_name}"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if joint_prim:
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            if drive_api:
                drive_api.GetTargetPositionAttr().Set(float(angle))


def print_com_report(result: dict):
    """Pretty-print the CoM analysis."""
    print("=" * 65)
    print("  BIPED ROBOT — CENTER OF MASS ANALYSIS")
    print("=" * 65)
    print()

    print(f"  {'Link':<35} {'Mass (kg)':>10}  {'CoM Local (x, y, z)':>25}")
    print(f"  {'-'*35} {'-'*10}  {'-'*25}")

    for d in result["per_link_coms"]:
        name = d["name"].split("/")[-1]  # short name
        com = d["com_local"]
        print(f"  {name:<35} {d['mass']:>10.4f}  ({com[0]:>7.4f}, {com[1]:>7.4f}, {com[2]:>7.4f})")

    print(f"  {'-'*35} {'-'*10}  {'-'*25}")
    print(f"  {'TOTAL':<35} {result['total_mass']:>10.4f}")
    print()

    oc = result["overall_com"]
    print(f"  OVERALL CoG (world frame):  ({oc[0]:.4f}, {oc[1]:.4f}, {oc[2]:.4f}) m")
    print(f"  Total mass:                  {result['total_mass']:.4f} kg")
    print()

    # Check CoG relative to base_link
    base_com = None
    for d in result["per_link_coms"]:
        if "base_link" in d["name"] and "base_link_1" not in d["name"]:
            base_com = d["com_world"]
            break

    if base_com:
        rel = tuple(oc[i] - base_com[i] for i in range(3))
        print(f"  CoG relative to base_link:   ({rel[0]:.4f}, {rel[1]:.4f}, {rel[2]:.4f}) m")
        print(f"  (Positive Z = below base_link center)")
    print()


def create_com_visualization(stage, overall_com: tuple):
    """Add a visual sphere at the overall CoG position in the USD stage.

    This creates a bright sphere so you can see the CoG in the viewport.
    """
    from pxr import UsdGeom, Sdf, Gf

    # Create a sphere at the CoG position
    sphere_path = Sdf.Path("/biped_robot/CoG_Visualization")
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.GetRadiusAttr().Set(0.015)  # 15mm radius

    # Color it bright yellow/red
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])

    # Position it
    xform = UsdGeom.XformCommonAPI(sphere.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*overall_com))

    print(f"  ✓ Added visualization sphere at: {overall_com}")
    print(f"    Path: {sphere_path}")
    print(f"    (Reload the stage in Isaac Sim to see it)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute robot CoG from USD")
    parser.add_argument(
        "--usd",
        type=str,
        default=None,
        help="Path to the robot USD file (default: biped_clean/biped_clean.usda)",
    )
    parser.add_argument(
        "--add-sphere",
        action="store_true",
        help="Add a visualization sphere at the overall CoG in the USD (and save)",
    )
    parser.add_argument(
        "--joints",
        type=str,
        default=None,
        help="Override joint angles as JSON string, e.g. '{\"r_knee_pitch_joint\": -0.5}'",
    )
    args = parser.parse_args()

    # Resolve default USD path
    if args.usd:
        usd_path = args.usd
    else:
        repo_root = Path(__file__).resolve().parent.parent
        usd_path = str(repo_root / "biped_clean" / "biped_clean.usda")

    if not Path(usd_path).exists():
        print(f"ERROR: USD file not found: {usd_path}")
        return

    print(f"\n  Reading: {usd_path}\n")

    # Parse optional joint overrides
    joint_positions = None
    if args.joints:
        import json
        joint_positions = json.loads(args.joints)

    # Compute CoM
    result = compute_overall_com(usd_path, joint_positions)
    print_com_report(result)

    # Optionally add visualization
    if args.add_sphere:
        from pxr import Usd
        stage = Usd.Stage.Open(usd_path)
        create_com_visualization(stage, result["overall_com"])
        stage.GetRootLayer().Save()
        print("  ✓ USD file saved with visualization sphere.\n")


if __name__ == "__main__":
    main()
