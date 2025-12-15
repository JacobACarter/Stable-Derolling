import os

folder1 = "input/X4K/ALL_GS_300/"
folder2 = "input/X4K/ALL_RS_300/"
output_file = "pairs.txt"

parent_dir = os.path.dirname(os.path.abspath(__file__))

def collect_files(folder, strip_pred=False):
    paths = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                name, ext = os.path.splitext(file)
                if strip_pred and name.endswith("_pred"):
                    name = name[:-5]  # remove "_pred"
                key = name + ext.lower()  # match extensions too
                paths[key] = os.path.join(root, file)
    return paths

files1 = collect_files(folder1, strip_pred=False)  # no "_pred" here
files2 = collect_files(folder2, strip_pred=True)   # remove "_pred"

common_files = sorted(set(files1.keys()) & set(files2.keys()))

with open(output_file, "w") as f:
    for filename in common_files:
        rel1 = os.path.relpath(files1[filename], parent_dir)
        rel2 = os.path.relpath(files2[filename], parent_dir)
        f.write(f"{rel1} {rel2}\n")

print(f"Paired {len(common_files)} files into {output_file}")