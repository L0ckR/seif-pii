"""Convert this presentation's SVG primitives to editable slide geometry.

Standard-library only. Supports SVG text, tspans, rectangles, circles, ellipses,
lines, orthogonal paths, simple CSS selectors and translated/scaled groups.
Unsupported visible content fails explicitly; there is no bitmap fallback.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
PROPS = {"fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity", "font-size", "font-family", "font-weight", "font-style", "text-anchor", "letter-spacing", "stroke-dasharray", "marker-end", "marker-start", "visibility", "display", "opacity"}
DEFAULT = {"fill": "#000000", "stroke": "none", "font-size": "16", "font-family": "Segoe UI", "text-anchor": "start", "font-weight": "400", "font-style": "normal"}


def number(value, default=0.0):
    if value is None:
        return default
    match = re.match(NUM, str(value))
    if not match:
        raise ValueError(f"Unsupported numeric value: {value!r}")
    return float(match.group())


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def declarations(value):
    return dict((k.strip(), v.strip()) for item in value.split(";") if ":" in item for k, v in [item.split(":", 1)])


def css_rules(root):
    result = []
    for element in root.iter():
        if tag(element) == "style":
            value = re.sub(r"/\*.*?\*/", "", "".join(element.itertext()), flags=re.S)
            for selectors, body in re.findall(r"([^{}]+)\{([^{}]+)\}", value):
                for selector in selectors.split(","):
                    result.append((selector.strip(), declarations(body)))
    return result


def matches(element, selector):
    if selector == "*":
        return True
    # Deliberately reject combinators; current deck uses only simple selectors.
    if any(c in selector for c in " >+~[:"):
        return False
    part = re.fullmatch(r"([\w-]+)?(?:#([\w-]+))?((?:\.[\w-]+)*)", selector)
    if not part:
        return False
    kind, ident, classes = part.groups()
    return ((not kind or tag(element) == kind)
            and (not ident or element.get("id") == ident)
            and set(c for c in classes.split(".") if c).issubset(element.get("class", "").split()))


def style_for(element, inherited, rules):
    result = {key: value for key, value in inherited.items() if key != "opacity"}
    result.update({key: val for key, val in element.attrib.items() if key in PROPS})
    matched = [(s.count("#") * 100 + s.count(".") * 10 + bool(re.match(r"^\w", s)), i, d)
               for i, (s, d) in enumerate(rules) if matches(element, s)]
    for _, _, vals in sorted(matched):
        result.update(vals)
    result.update(declarations(element.get("style", "")))
    return result


def multiply(a, b):
    aa, ab, ac, ad, ae, af = a
    ba, bb, bc, bd, be, bf = b
    return (aa*ba+ac*bb, ab*ba+ad*bb, aa*bc+ac*bd, ab*bc+ad*bd, aa*be+ac*bf+ae, ab*be+ad*bf+af)


def transform(value):
    result = (1, 0, 0, 1, 0, 0)
    for name, raw in re.findall(r"([A-Za-z]+)\s*\(([^)]+)\)", value):
        n = list(map(float, re.findall(NUM, raw)))
        if name == "translate":
            m = (1, 0, 0, 1, n[0], n[1] if len(n) > 1 else 0)
        elif name == "scale":
            m = (n[0], 0, 0, n[1] if len(n) > 1 else n[0], 0, 0)
        elif name == "matrix" and len(n) == 6:
            m = tuple(n)
        else:
            raise ValueError(f"Unsupported SVG transform: {name}")
        result = multiply(result, m)
    if abs(result[1]) > 1e-9 or abs(result[2]) > 1e-9:
        raise ValueError("Rotated/skewed groups need an explicit conversion, not a raster fallback")
    return result


def xy(m, x, y):
    return m[0]*x+m[2]*y+m[4], m[1]*x+m[3]*y+m[5]


def color(value, opacity=1.0):
    if value in (None, "none", "transparent"):
        return None
    value = value.strip()
    alpha = 1.0
    if value.startswith("#"):
        value = value[1:]
        if len(value) in (3, 4):
            value = "".join(c*2 for c in value)
        if len(value) == 8:
            alpha = int(value[6:], 16)/255
            value = value[:6]
        if len(value) != 6:
            raise ValueError(f"Unsupported SVG color: {value}")
    elif value.startswith("rgb"):
        nums = re.findall(NUM, value)
        value = "".join(f"{int(float(x)):02x}" for x in nums[:3])
        if len(nums) == 4:
            alpha = float(nums[3])
    elif value in {"white", "black"}:
        value = "ffffff" if value == "white" else "000000"
    else:
        raise ValueError(f"Unsupported SVG color: {value}")
    return {"color": value.upper(), "transparency": round(100*(1-max(0, min(1, opacity*alpha))), 4)}


class FontMetrics:
    """Read advance widths and cmap from the same local TrueType fonts as PPT."""
    def __init__(self, path):
        self.data = path.read_bytes()
        data = self.data
        count = struct.unpack_from(">H", data, 4)[0]
        self.tables = {}
        for i in range(count):
            name, _, offset, length = struct.unpack_from(">4sIII", data, 12+16*i)
            self.tables[name.decode("ascii")] = (offset, length)
        self.units = struct.unpack_from(">H", data, self.tables["head"][0]+18)[0]
        hhea = self.tables["hhea"][0]
        self.ascent, self.descent, self.gap = struct.unpack_from(">hhh", data, hhea+4)
        self.num_metrics = struct.unpack_from(">H", data, hhea+34)[0]
        base = self.tables["hmtx"][0]
        self.widths = [struct.unpack_from(">H", data, base+4*i)[0] for i in range(self.num_metrics)]
        cmap = self.tables["cmap"][0]
        choices = []
        for i in range(struct.unpack_from(">H", data, cmap+2)[0]):
            platform, encoding, offset = struct.unpack_from(">HHI", data, cmap+4+8*i)
            sub = cmap+offset
            fmt = struct.unpack_from(">H", data, sub)[0]
            if platform in (0, 3) and fmt in (4, 12):
                choices.append((fmt, platform == 3, encoding, sub))
        _, _, _, self.cmap = max(choices)
        self.format = struct.unpack_from(">H", data, self.cmap)[0]
        self.groups = None
        if self.format == 12:
            count = struct.unpack_from(">I", data, self.cmap+12)[0]
            self.groups = [struct.unpack_from(">III", data, self.cmap+16+i*12) for i in range(count)]
            self.starts = [g[0] for g in self.groups]

    def glyph(self, cp):
        data, sub = self.data, self.cmap
        if self.format == 12:
            i = bisect.bisect_right(self.starts, cp)-1
            if i >= 0 and self.groups[i][0] <= cp <= self.groups[i][1]:
                return self.groups[i][2]+cp-self.groups[i][0]
            return 0
        if cp > 65535:
            return 0
        count = struct.unpack_from(">H", data, sub+6)[0]//2
        ends = sub+14
        starts, deltas, ranges = ends+2*count+2, ends+4*count+2, ends+6*count+2
        for i in range(count):
            end = struct.unpack_from(">H", data, ends+2*i)[0]
            if cp > end:
                continue
            begin = struct.unpack_from(">H", data, starts+2*i)[0]
            if cp < begin:
                return 0
            delta = struct.unpack_from(">h", data, deltas+2*i)[0]
            offset = struct.unpack_from(">H", data, ranges+2*i)[0]
            if not offset:
                return (cp+delta) % 65536
            gid = struct.unpack_from(">H", data, ranges+2*i+offset+2*(cp-begin))[0]
            return (gid+delta) % 65536 if gid else 0
        return 0

    def width(self, text, size, spacing=0):
        glyphs = [self.glyph(ord(c)) for c in text]
        return sum(self.widths[min(g, self.num_metrics-1)] for g in glyphs)*size/self.units + max(0,len(text)-1)*spacing


@lru_cache(maxsize=12)
def get_font(family, bold, italic):
    mono = any(s in family.lower() for s in ("mono", "consolas"))
    serif = "georgia" in family.lower() or ("serif" in family.lower() and "sans-serif" not in family.lower())
    face = "Consolas" if mono else "Georgia" if serif else "Segoe UI"
    names = {"Segoe UI": ("segoeui", "segoeuib", "segoeuii", "segoeuiz"), "Georgia": ("georgia", "georgiab", "georgiai", "georgiaz"), "Consolas": ("consola", "consolab", "consolai", "consolaz")}
    filename = names[face][int(bold)+2*int(italic)]+".ttf"
    candidates = [Path("/mnt/c/Windows/Fonts")/filename, Path("C:/Windows/Fonts")/filename]
    for path in candidates:
        if path.is_file():
            return face, FontMetrics(path)
    raise FileNotFoundError(f"Local TTF required for reproducible editable text width: {filename}")


def segments(value):
    tokens = re.findall(r"[A-Za-z]|"+NUM, value)
    pos, start, command, out, i = (0., 0.), (0., 0.), None, [], 0
    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        if command is None or command.upper() not in ("M", "L", "H", "V", "Z"):
            raise ValueError(f"Unsupported SVG path command: {command}")
        kind, relative = command.upper(), command.islower()
        if kind == "Z":
            out.append((pos, start));pos=start;command=None
            continue
        n = 1 if kind in ("H", "V") else 2
        vals = list(map(float, tokens[i:i+n]));i += n
        if kind == "H":
            dest = (vals[0]+(pos[0] if relative else 0), pos[1])
        elif kind == "V":
            dest = (pos[0], vals[0]+(pos[1] if relative else 0))
        else:
            dest = (vals[0]+(pos[0] if relative else 0), vals[1]+(pos[1] if relative else 0))
        if kind == "M":
            start = dest;command="l" if relative else "L"
        else:
            out.append((pos, dest))
        pos = dest
    return out


def convert_file(path, metadata=None):
    root = ET.parse(path).getroot()
    rules, items, warnings = css_rules(root), [], []
    view = list(map(float, root.get("viewBox", "0 0 1600 900").split()))
    if view != [0., 0., 1600., 900.]:
        raise ValueError(f"Expected deck canvas 1600x900: {path}")

    def paint(style, opacity, m):
        return {"fill": color(style.get("fill"), opacity*number(style.get("fill-opacity"),1)),
                "line": color(style.get("stroke"), opacity*number(style.get("stroke-opacity"),1)),
                "lineWidth": number(style.get("stroke-width"),1)*math.sqrt(abs(m[0]*m[3])),
                "dash": style.get("stroke-dasharray") not in (None, "none", "0")}

    def text_item(text, x, y, style, opacity, m, element):
        if not text:
            return 0
        size = number(style.get("font-size"),16)*abs(m[3])
        weight = style.get("font-weight", "400")
        bold = weight in ("bold", "bolder") or (weight.isdigit() and int(weight) >= 600)
        italic = style.get("font-style") in ("italic", "oblique")
        face, metrics = get_font(style.get("font-family", ""), bold, italic)
        spacing = number(style.get("letter-spacing"),0)*abs(m[0]) if style.get("letter-spacing") != "normal" else 0
        width = metrics.width(text,size,spacing)*1.035+3
        maxwidth = number(element.get("data-max-width"),0)*abs(m[0])
        if maxwidth and width > maxwidth+2:
            warnings.append({"type":"text_maxwidth", "text":text, "measured_px":round(width,2), "max_px":maxwidth})
        ax, ay = xy(m,x,y)
        anchor = style.get("text-anchor", "start")
        if anchor == "middle":ax -= width/2
        if anchor == "end":ax -= width
        ascent = metrics.ascent/metrics.units*size
        height = (metrics.ascent-metrics.descent+metrics.gap)/metrics.units*size+2
        items.append({"kind":"text", "text":text, "x":ax, "y":ay-ascent, "w":width, "h":height,
                      "fontSize":size, "fontFace":face, "bold":bold, "italic":italic, "spacing":spacing,
                      "align":{"start":"left","middle":"center","end":"right"}[anchor],
                      "fill":color(style.get("fill"),opacity*number(style.get("fill-opacity"),1)),
                      "baseline":ay, "svgFontFamily":style.get("font-family")})
        return metrics.width(text,number(style.get("font-size"),16),spacing/max(abs(m[0]),1e-12))

    def walk_text(element, style, opacity, m, x=None, y=None):
        x = number(element.get("x"),0 if x is None else x)+number(element.get("dx"),0)
        y = number(element.get("y"),0 if y is None else y)+number(element.get("dy"),0)
        x += text_item(element.text or "",x,y,style,opacity,m,element)
        for child in element:
            if tag(child) != "tspan":
                raise ValueError("Only tspan is supported inside SVG text")
            child_style = style_for(child,style,rules)
            child_opacity = opacity*number(child_style.get("opacity"),1)
            x,y = walk_text(child,child_style,child_opacity,m,x,y)
            x += text_item(child.tail or "",x,y,style,opacity,m,element)
        return x,y

    def walk(element, inherited, parent_matrix, parent_opacity):
        kind = tag(element)
        if kind in {"defs","title","desc","style","metadata"}:
            return
        style = style_for(element,inherited,rules)
        opacity = parent_opacity*number(style.get("opacity"),1)
        if style.get("display") == "none" or style.get("visibility") == "hidden" or opacity == 0:
            return
        m = multiply(parent_matrix,transform(element.get("transform","")))
        if kind in {"svg","g","a"}:
            for child in element:walk(child,style,m,opacity)
            return
        p = paint(style,opacity,m)
        if kind == "text":
            walk_text(element,style,opacity,m)
        elif kind == "rect":
            x,y = xy(m,number(element.get("x")),number(element.get("y")))
            w,h = number(element.get("width"))*m[0],number(element.get("height"))*m[3]
            items.append({"kind":"rect","x":x,"y":y,"w":w,"h":h,"radius":number(element.get("rx"))*abs(m[0]),**p})
        elif kind in {"circle","ellipse"}:
            cx,cy = xy(m,number(element.get("cx")),number(element.get("cy")))
            rx = number(element.get("rx",element.get("r")))*abs(m[0]); ry = number(element.get("ry",element.get("r")))*abs(m[3])
            items.append({"kind":"ellipse","x":cx-rx,"y":cy-ry,"w":2*rx,"h":2*ry,**p})
        elif kind in {"line","path","polyline"}:
            if kind == "path":
                if style.get("fill") not in ("none","transparent"):
                    raise ValueError("Filled custom path needs explicit native geometry support")
                pairs = segments(element.get("d",""))
            elif kind == "polyline":
                nums = list(map(float,re.findall(NUM,element.get("points","")))); pts=list(zip(nums[::2],nums[1::2]));pairs=list(zip(pts,pts[1:]))
            else:
                pairs = [((number(element.get("x1")),number(element.get("y1"))),(number(element.get("x2")),number(element.get("y2"))))]
            for i,(a,b) in enumerate(pairs):
                x1,y1 = xy(m,*a);x2,y2 = xy(m,*b)
                if x1 == x2 and y1 == y2:continue
                items.append({"kind":"line","x1":x1,"y1":y1,"x2":x2,"y2":y2,**p,
                              "arrowEnd":i==len(pairs)-1 and style.get("marker-end") not in (None,"none"),
                              "arrowStart":i==0 and style.get("marker-start") not in (None,"none")})
        else:
            raise ValueError(f"Unsupported visible SVG element: {kind} in {path}")
    walk(root,DEFAULT,(1,0,0,1,0,0),1)
    for i,item in enumerate(items):
        if item["kind"] == "line":
            x1,x2=sorted((item["x1"],item["x2"])); y1,y2=sorted((item["y1"],item["y2"]))
        else:x1,y1,x2,y2=item["x"],item["y"],item["x"]+item["w"],item["y"]+item["h"]
        if min(x1,y1) < -0.1 or x2 > 1600.1 or y2 > 900.1:
            warnings.append({"type":"canvas_bounds","item":i,"kind":item["kind"],"text":item.get("text"),"bounds":[x1,y1,x2,y2]})
    return {**(metadata or {}),"file":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"items":items,"warnings":warnings,"counts":dict(Counter(i["kind"] for i in items))}


def validate_pptx(path):
    ns={"p":"http://schemas.openxmlformats.org/presentationml/2006/main","a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
    with zipfile.ZipFile(path) as z:
        files=sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml",n)),key=lambda p:int(re.search(r"slide(\d+)",p).group(1)))
        counts=[];problems=[]
        for file in files:
            root=ET.fromstring(z.read(file)); texts=root.findall('.//a:t',ns); shapes=root.findall('.//p:sp',ns); pics=root.findall('.//p:pic',ns)
            counts.append({"file":file,"editable_text_runs":len(texts),"native_shapes":len(shapes),"pictures":len(pics),"text_sha256":hashlib.sha256("\0".join(t.text or "" for t in texts).encode()).hexdigest()})
            for xf in root.findall('.//a:xfrm',ns):
                off,ext=xf.find('a:off',ns),xf.find('a:ext',ns)
                if off is None or ext is None:continue
                x,y,cx,cy=[int(v) for v in [off.get('x'),off.get('y'),ext.get('cx'),ext.get('cy')]]
                if x < -762 or y < -762 or x+cx > 12192000+762 or y+cy > 6858000+762:
                    problems.append({"file":file,"bounds_emu":[x,y,cx,cy]})
        notes=[n for n in z.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml",n)]
    return {"slide_count":len(files),"notes_count":len(notes),"slides":counts,"editable_text_runs":sum(x['editable_text_runs'] for x in counts),"native_shapes":sum(x['native_shapes'] for x in counts),"picture_count":sum(x['pictures'] for x in counts),"canvas_bounds_errors":problems}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slides',type=Path,default=HERE/'slides')
    parser.add_argument('--deck',type=Path,default=HERE/'deck.json')
    parser.add_argument('--output',type=Path,default=HERE/'pptx-scene.json')
    parser.add_argument('--validate-pptx',type=Path)
    args=parser.parse_args()
    if args.validate_pptx:
        print(json.dumps(validate_pptx(args.validate_pptx),ensure_ascii=False,indent=2));return
    paths=sorted(args.slides.glob('*.svg'))
    if not paths:raise SystemExit(f'No SVG slides found in {args.slides}')
    metadata=json.loads(args.deck.read_text()) if args.deck.is_file() else [{} for _ in paths]
    if isinstance(metadata,dict):metadata=metadata['slides']
    if len(metadata)!=len(paths):raise ValueError('Slide count does not match deck metadata')
    scenes=[convert_file(path,meta) for path,meta in zip(paths,metadata,strict=True)]
    result={"canvas":{"width":1600,"height":900,"pixels_per_inch":120},"slides":scenes,
            "conversion":{"mode":"native editable text and shapes; no image fallback","font_mapping":{"sans":"Segoe UI","serif":"Georgia","mono":"Consolas"},"limitations":["Arbitrary Bézier paths, skew and rotated groups fail explicitly.","Dash arrays are mapped to native PowerPoint dash presets.","Text uses local TrueType advances; applications may render font baselines slightly differently.","Tspan chunks are separate editable text boxes; current slides have no tspans."]}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({"slides":len(scenes),"elements":sum(len(s['items']) for s in scenes),"warnings":sum(len(s['warnings']) for s in scenes),"output":str(args.output)},ensure_ascii=False))

if __name__=='__main__':main()
