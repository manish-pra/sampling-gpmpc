from xml.etree import ElementTree as ET

urdf_file = "/home/manish/work/MPC_Dyn/urdf2casadi/examples/urdf/ur5_mod.urdf"
output_file = "/home/manish/work/MPC_Dyn/urdf2casadi/examples/urdf/ur5_mod_visual.urdf"

tree = ET.parse(urdf_file)
root = tree.getroot()

for link in root.findall("link"):
    if link.find("visual") is None:
        visual = ET.SubElement(link, "visual")
        geom = ET.SubElement(visual, "geometry")
        cyl = ET.SubElement(geom, "cylinder", length="0.1", radius="0.02")
        ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        mat = ET.SubElement(visual, "material", name="gray")
        ET.SubElement(mat, "color", rgba="0.5 0.5 0.5 1.0")

tree.write(output_file)
print(f"Patched URDF written to {output_file}")