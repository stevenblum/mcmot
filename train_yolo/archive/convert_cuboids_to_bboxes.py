import xml.etree.ElementTree as ET
import sys
import os

def cuboid_to_bbox(cuboid_elem):
    # Extract all x/y coordinates from cuboid attributes
    point_attrs = [
        'xtl1', 'ytl1', 'xbl1', 'ybl1', 'xtr1', 'ytr1', 'xbr1', 'ybr1',
        'xtl2', 'ytl2', 'xbl2', 'ybl2', 'xtr2', 'ytr2', 'xbr2', 'ybr2'
    ]
    coords = []
    for i in range(0, len(point_attrs), 2):
        x = cuboid_elem.get(point_attrs[i])
        y = cuboid_elem.get(point_attrs[i+1])
        if x is not None and y is not None:
            coords.append((float(x), float(y)))
    if not coords:
        return None
    xs, ys = zip(*coords)
    return min(xs), min(ys), max(xs), max(ys)

def convert_xml(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"Error: Input XML file not found: {input_path}")
        print("Hint: The input XML should be something like ./annotations/annotations.xml")
        sys.exit(1)
    tree = ET.parse(input_path)
    root = tree.getroot()
    if root.tag != 'annotations':
        print("Not a CVAT XML file.")
        return

    for image_elem in root.findall('image'):
        # Remove existing box elements if any
        for box in image_elem.findall('box'):
            image_elem.remove(box)
        # Convert cuboids to boxes
        for cuboid_elem in list(image_elem.findall('cuboid')):
            bbox = cuboid_to_bbox(cuboid_elem)
            if bbox is None:
                continue
            xtl, ytl, xbr, ybr = bbox
            # Determine new label
            orig_label = cuboid_elem.get('label', '')
            new_label = orig_label
            if orig_label == 'part':
                shape = None
                color = None
                for attr_elem in cuboid_elem.findall('attribute'):
                    if attr_elem.get('name') == 'shape':
                        shape = attr_elem.text.strip().lower()
                    elif attr_elem.get('name') == 'color':
                        color = attr_elem.text.strip().lower()
                if shape and color:
                    new_label = f"{color}_{shape}"
            # Create new box element
            box_elem = ET.Element('box')
            box_elem.set('label', new_label)
            box_elem.set('xtl', str(xtl))
            box_elem.set('ytl', str(ytl))
            box_elem.set('xbr', str(xbr))
            box_elem.set('ybr', str(ybr))
            # Copy over shape/color attributes if present
            for attr_elem in cuboid_elem.findall('attribute'):
                box_elem.append(attr_elem)
            image_elem.append(box_elem)
        # Remove cuboid elements
        for cuboid_elem in list(image_elem.findall('cuboid')):
            image_elem.remove(cuboid_elem)

    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Converted XML saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_cuboids_to_bboxes.py input.xml output.xml")
        print("Example: python convert_cuboids_to_bboxes.py ./annotations/annotations.xml output.xml")
        sys.exit(1)
    convert_xml(sys.argv[1], sys.argv[2])
