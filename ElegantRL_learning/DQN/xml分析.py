import xml.etree.ElementTree as ET
tree = ET.parse('tripinfo.xml')
root = tree.getroot()
print(root)
for child in root[0:1]:
    print(child.tag,child.attrib)
    # print(len(child.attrib))
    print(child.attrib['id'])