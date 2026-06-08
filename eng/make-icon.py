"""Generates the Hnsw.Net package icon (original artwork).

A rounded-square gradient badge containing a small-world graph: a highlighted
query node connected by edges to its nearest-neighbor nodes. Represents HNSW
approximate-nearest-neighbor search. Intentionally distinct from any upstream logo.
"""
from PIL import Image, ImageDraw

S = 512  # supersample, downscaled at the end
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Vertical gradient background: deep navy -> cyan
top = (27, 42, 91)       # #1B2A5B
bot = (20, 184, 198)     # #14B8C6
for y in range(S):
    t = y / (S - 1)
    r = round(top[0] + (bot[0] - top[0]) * t)
    g = round(top[1] + (bot[1] - top[1]) * t)
    b = round(top[2] + (bot[2] - top[2]) * t)
    draw.line([(0, y), (S, y)], fill=(r, g, b, 255))

# Rounded-square mask
radius = int(S * 0.22)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=255)
img.putalpha(mask)

draw = ImageDraw.Draw(img)

# Graph nodes in normalized coordinates (x, y)
query = (0.50, 0.52)
neighbors = [(0.22, 0.30), (0.78, 0.28), (0.26, 0.74), (0.76, 0.72), (0.50, 0.20)]
# A couple of outer nodes (not directly linked to query) for small-world feel
outer = [(0.16, 0.54), (0.84, 0.52)]


def px(p):
    return (p[0] * S, p[1] * S)


white = (255, 255, 255, 255)
edge = (255, 255, 255, 150)
accent = (245, 197, 66, 255)  # amber query highlight

# Edges: query -> each neighbor
ew = max(2, int(S * 0.012))
for n in neighbors:
    draw.line([px(query), px(n)], fill=edge, width=ew)
# A few neighbor-to-neighbor / neighbor-to-outer edges
for a, b in [(neighbors[0], outer[0]), (neighbors[2], outer[0]),
             (neighbors[1], outer[1]), (neighbors[3], outer[1]),
             (neighbors[4], neighbors[1])]:
    draw.line([px(a), px(b)], fill=(255, 255, 255, 90), width=max(2, int(S * 0.009)))


def node(p, rad, fill):
    x, y = px(p)
    draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=fill)


# Outer nodes (smaller, dimmer)
for p in outer:
    node(p, int(S * 0.035), (255, 255, 255, 190))
# Neighbor nodes
for p in neighbors:
    node(p, int(S * 0.05), white)
# Query node (largest, accent ring + accent fill)
node(query, int(S * 0.085), white)
node(query, int(S * 0.062), accent)

out = img.resize((128, 128), Image.LANCZOS)
out.save(r"C:\src\ericstj\Hnsw.Net\eng\icon.png", "PNG")
print("wrote eng/icon.png")
