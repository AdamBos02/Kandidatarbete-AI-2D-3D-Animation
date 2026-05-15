

#export av mediapipes rörelsedata till csv-fil
import csv
from pathlib import Path


OUTPUT_CSV = r"C:\Users\tjede\Downloads\indianhopp_victor.csv"

NAME_PREFIX = ""

empties = [o for o in bpy.data.objects
           if o.type == "EMPTY" and o.name.startswith(NAME_PREFIX)]
empties.sort(key=lambda o: o.name)




scene = bpy.context.scene
f_start = scene.frame_start
f_end   = scene.frame_end
fps     = scene.render.fps / scene.render.fps_base
print(f"\nExporterar frames {f_start}–{f_end} ({fps:.3f} fps) till {OUTPUT_CSV}")

header = ["frame"]
for o in empties:
    header += [f"{o.name}_x", f"{o.name}_y", f"{o.name}_z"]

orig_frame = scene.frame_current

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"# fps={fps}"])
    writer.writerow(header)
    for fi in range(f_start, f_end + 1):
        scene.frame_set(fi)
        row = [fi]
        for o in empties:
            
            loc = o.matrix_world.translation
            row += [loc.x, loc.y, loc.z]
        writer.writerow(row)

scene.frame_set(orig_frame)
