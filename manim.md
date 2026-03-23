# Manim CE intermediate reference for Claude Code

**This document is a comprehensive reference for Manim Community Edition (v0.17–v0.20+) targeting intermediate-level usage across mathematical animations, algorithm visualizations, medical imaging pipelines, and scientific charts.** It is structured so Claude Code can directly reference patterns, API signatures, and working snippets when generating Manim code. All class names, methods, and parameters have been verified against official Manim CE documentation and source code.

---

## Core scene structure and configuration

### Scene lifecycle

```python
from manim import *

class MyScene(Scene):
    def setup(self):
        """Called before construct(). Use instead of __init__."""
        pass

    def construct(self):
        """Main animation logic."""
        circle = Circle()
        self.play(Write(circle))       # animated addition
        self.wait(1)                    # pause 1 second
        self.remove(circle)            # instant removal (no animation)

    def tear_down(self):
        """Called after construct(). Cleanup."""
        pass
```

**Key scene methods:**

| Method | Purpose |
|--------|---------|
| `self.play(*animations, run_time=1)` | Render and play animations |
| `self.add(*mobjects)` | Instantly display mobjects (no animation); z-order = add order |
| `self.remove(*mobjects)` | Instantly remove from scene |
| `self.clear()` | Remove all mobjects |
| `self.wait(duration=1)` | Pause; required for video output (without it, renders static PNG) |
| `self.next_section(name, skip_animations=False)` | Logical section break (use `--save_sections` CLI flag) |
| `self.add_foreground_mobjects(*mobs)` | Always-on-top layer |

**`self.play(FadeIn(mob))` vs `self.add(mob)`**: `play()` creates an animated introduction over `run_time` seconds and auto-adds the mob to the scene. `add()` is instant, appearing on the next rendered frame with no animation. Use `add()` for background elements or pre-building scenes before animating.

### Configuration

```bash
# CLI quality presets:
manim render -ql scene.py MyScene   # 854×480 @ 15fps (low)
manim render -qm scene.py MyScene   # 1280×720 @ 30fps (medium)
manim render -qh scene.py MyScene   # 1920×1080 @ 60fps (high)
manim render -qk scene.py MyScene   # 3840×2160 @ 60fps (4K)

# Other useful flags:
#   -p             preview after render
#   -s             render last frame as PNG only
#   --format gif   GIF output
#   -t             transparent background
#   -r 1000,500    custom resolution
#   --fps 24       custom framerate
#   -a             render all scenes in file
#   -c WHITE       background color
```

**manim.cfg file** (place in same directory as script):
```ini
[CLI]
background_color = WHITE
quality = medium_quality
frame_rate = 30
pixel_width = 1920
pixel_height = 1080
```

**Programmatic config:**
```python
from manim import config
config.background_color = "#1e1e2e"
config.frame_width = 14.222   # logical units (default)
config.frame_height = 8.0     # logical units (default)
config.pixel_width = 1920
config.pixel_height = 1080
config.format = "mp4"         # "mp4", "gif", "mov", "png"
config.transparent = True     # alpha channel

# Temporary config override:
with tempconfig({"background_color": WHITE, "frame_rate": 15}):
    scene.render()
```

**Precedence** (lowest→highest): library defaults → user-wide config → folder `manim.cfg` → CLI flags → programmatic `config.xxx`.

---

## MathTex, Tex, and Text objects

### Class distinctions

**`Text`** — Pango renderer (not LaTeX). Supports system fonts.
```python
Text(text, font=None, font_size=DEFAULT_FONT_SIZE, color=WHITE,
     t2c=None, t2f=None, t2g=None, t2s=None, t2w=None)
# t2c={"Hello": BLUE}  text-to-color dict
# t2f={"Hello": "Arial"}  text-to-font
# t2w={"Hello": BOLD}  text-to-weight
```
Use for: plain text, titles, labels, non-Latin scripts. Cannot use LaTeX commands.

**`MathTex`** — LaTeX math mode (`align*` environment).
```python
MathTex(*tex_strings, arg_separator=' ', substrings_to_isolate=None,
        tex_to_color_map=None, tex_environment='align*', font_size=DEFAULT_FONT_SIZE)
```
Use for: all mathematical formulas. Each string argument becomes a separate submobject.

**`Tex`** — LaTeX text mode (`center` environment). Subclass of MathTex.
```python
Tex(*tex_strings, arg_separator='', tex_environment='center')
```
Use for: mixed text and math (inline math via `$...$`).

**Key rule**: `MathTex("formula")` ≈ `Tex("$$formula$$")`. Always use raw strings: `r"\frac{1}{2}"`.

### Double-brace syntax for submobject splitting (v0.17+)
```python
eq = MathTex(r"{{ a^2 }} + {{ b^2 }} = {{ c^2 }}")
# Creates 5 submobjects: "a^2", "+", "b^2", "=", "c^2"
# Required for TransformMatchingTex to work cleanly
```

### Key methods on MathTex/Tex
```python
.set_color_by_tex("x", RED)                     # color submobjects matching "x"
.set_color_by_tex_to_color_map({"x": RED, "y": BLUE})
.get_part_by_tex("x")                           # first matching submobject
.get_parts_by_tex("x")                          # VGroup of all matches
```

**Gotcha**: `set_color_by_tex` requires exact match against a submobject's tex_string. If you pass a single long string, it's one submobject — use `substrings_to_isolate` or `{{ }}` to split. Use `index_labels(tex[0])` to debug glyph indices.

### Custom LaTeX packages
```python
from manim import TexTemplate
template = TexTemplate()
template.add_to_preamble(r"\usepackage{physics}")
eq = MathTex(r"\bra{\psi}\ket{\phi}", tex_template=template)
```

Default packages included: `amsmath`, `amssymb`, `babel[english]`.

---

## Formula transformations and step-by-step derivations

### TransformMatchingTex
```python
TransformMatchingTex(source, target, transform_mismatches=False,
                     fade_transform_mismatches=False, key_map=None, **kwargs)
```
Matches submobjects by identical `tex_string`. Both source and target must have properly split submobjects via `{{ }}` or multiple string args.

```python
eq1 = MathTex("{{x}}+{{y}}={{4}}")
eq2 = MathTex("{{x}}={{4}}-{{y}}")
self.play(TransformMatchingTex(eq1, eq2, key_map={"+": "-"}), run_time=2)
# key_map morphs "+" into "-" instead of fade
```

### TransformMatchingShapes
```python
TransformMatchingShapes(source, target, transform_mismatches=False,
                        fade_transform_mismatches=False, **kwargs)
```
Matches by geometric shape hash, not tex_string. Works with any Mobject. Use when you don't want to worry about submobject splitting.

### Other transforms
- **`Transform(source, target)`** — morphs source; source identity preserved in scene graph
- **`ReplacementTransform(source, target)`** — replaces source with target in scene graph
- **`TransformFromCopy(source, target)`** — keeps source, animates a copy into target

### Pattern: vertical derivation with TransformFromCopy
```python
lines = VGroup(
    MathTex(r"A^2 + B^2 = C^2"),
    MathTex(r"A^2 = C^2 - B^2"),
    MathTex(r"A = \sqrt{C^2 - B^2}"),
)
lines.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

self.play(Write(lines[0]))
self.play(TransformFromCopy(lines[0], lines[1]))
self.play(TransformFromCopy(lines[1], lines[2]))
```

### Pattern: shift equation up, write next step below
```python
self.play(Write(equation))
self.play(equation.animate.shift(UP))
equation2 = MathTex(r"x = 4").next_to(equation, DOWN)
self.play(Write(equation2))
```

### Pattern: color-coded derivation
```python
to_isolate = ["A", "B", "C"]
lines = VGroup(
    MathTex("A^2", "+", "B^2", "=", "C^2", substrings_to_isolate=to_isolate),
    MathTex("A^2", "=", "C^2", "-", "B^2", substrings_to_isolate=to_isolate),
)
lines.arrange(DOWN, buff=LARGE_BUFF)
for line in lines:
    line.set_color_by_tex_to_color_map({"A": BLUE, "B": TEAL, "C": GREEN})
```

### Multi-line aligned equations
```python
derivation = MathTex(
    r"f(x) &= 3 + 2 + 1 \\",
    r"&= 5 + 1 \\",
    r"&= 6"
)
# The & aligns at equals sign (standard LaTeX align*)
```

### Common statistical formulas
```python
gaussian = MathTex(
    r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}",
    r"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}"
)
psnr = MathTex(r"\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)")
```

---

## Formula highlighting and indication animations

### Indicate (non-destructive pulse)
```python
Indicate(mobject, scale_factor=1.2, color=YELLOW, rate_func=there_and_back)
# Temporarily scales and recolors, then returns to original
self.play(Indicate(equation[2], color=RED, scale_factor=1.5))
```

### Circumscribe (outline flash)
```python
Circumscribe(mobject, shape=Rectangle, fade_in=False, fade_out=False,
             time_width=0.3, buff=SMALL_BUFF, color=YELLOW)
self.play(Circumscribe(equation[1], Circle))       # circle flash
self.play(Circumscribe(equation[1], fade_out=True)) # fades out
```

### Flash (burst effect)
```python
Flash(point_or_mobject, line_length=0.2, num_lines=12, flash_radius=0.1, color=YELLOW)
self.play(Flash(equation[3], line_length=0.5, num_lines=20, color=RED))
```

### FocusOn (spotlight)
```python
FocusOn(focus_point, opacity=0.2, color=GREY, run_time=2)
```

### SurroundingRectangle (persistent highlight)
```python
SurroundingRectangle(*mobjects, color=YELLOW, buff=0.1, corner_radius=0.0)

box1 = SurroundingRectangle(eq[1], buff=0.1, color=YELLOW)
box2 = SurroundingRectangle(eq[3], buff=0.1, color=BLUE)
self.play(Create(box1))
self.play(ReplacementTransform(box1, box2))  # move highlight between terms
```

### Brace annotation
```python
brace = Brace(mobject, direction=DOWN, buff=0.2)
brace_text = brace.get_text("Width")      # returns Tex
brace_tex = brace.get_tex("x - x_1")      # returns MathTex
self.play(GrowFromCenter(brace), FadeIn(brace_text))
```

### Other indication animations
- **`ApplyWave(mobject, direction=UP, amplitude=0.2)`** — wave distortion
- **`Wiggle(mobject, scale_value=1.1, rotation_angle=0.01*TAU, n_wiggles=6)`** — wiggle in place
- **`ShowPassingFlash(vmobject, time_width=0.1)`** — flash travels along path

---

## Positioning, alignment, and grouping

### Core positioning methods
```python
mob.move_to(point_or_mobject)                    # absolute position
mob.move_to(other, aligned_edge=LEFT)            # align left edges
mob.next_to(mobject, DOWN, buff=0.25)            # relative position
mob.next_to(mobject, RIGHT, buff=0.5, aligned_edge=UP)
mob.align_to(other, LEFT)                        # align specific edges
mob.shift(UP * 2 + RIGHT * 3)                    # relative displacement
mob.to_edge(UP, buff=MED_LARGE_BUFF)             # snap to frame edge
mob.to_corner(UL)                                # snap to corner
```

### Position query methods
```python
mob.get_center()    mob.get_top()     mob.get_bottom()
mob.get_left()      mob.get_right()   mob.get_corner(UL)
mob.get_start()     mob.get_end()     mob.get_edge_center(UP)
```

### VGroup — grouping and arranging
```python
group = VGroup(eq1, eq2, eq3)
group = VGroup(*[MathTex(f"x_{i}") for i in range(5)])

# Arrange in a line
group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)

# Arrange in a grid
boxes = VGroup(*[Square() for _ in range(12)])
boxes.arrange_in_grid(rows=3, cols=4, buff=0.25)
# Advanced: buff=(row_gap, col_gap), col_alignments="lccr", flow_order="dr"
```

**`Group` vs `VGroup`**: `VGroup` only accepts `VMobject` subclasses (Text, Tex, shapes). `Group` accepts any `Mobject` including `ImageMobject`. **Always use `Group` when mixing ImageMobject with other types.**

---

## Animation chaining and sequencing

### AnimationGroup — simultaneous (default)
```python
AnimationGroup(*animations, lag_ratio=0, run_time=None, rate_func=linear)
# lag_ratio=0: all simultaneous
# lag_ratio=1: fully sequential (like Succession)
# lag_ratio=0.5: each starts when previous is 50% done
```

### Succession — strictly sequential
```python
Succession(*animations, lag_ratio=1.0)
# Total run_time distributed across animations proportionally
self.play(Succession(
    FadeIn(obj1), obj1.animate.shift(RIGHT), FadeOut(obj1),
    run_time=3
))
```

### LaggedStart — staggered start (default lag_ratio=0.05)
```python
LaggedStart(*animations, lag_ratio=0.05)
self.play(LaggedStart(
    *[FadeIn(cell) for cell in grid_cells],
    lag_ratio=0.1, run_time=3
))
```

### LaggedStartMap — apply animation class to each submobject
```python
LaggedStartMap(animation_class, mobject, lag_ratio=0.05, run_time=2)
self.play(LaggedStartMap(FadeIn, dots_group, lag_ratio=0.1))
```

**Combining**: `Succession(AnimationGroup(a1, a2), AnimationGroup(a3, a4))` plays two groups sequentially, each group's animations playing simultaneously.

---

## Algorithm step-by-step walkthroughs

### Table classes for 2D arrays

```python
# Base Table class
Table(table_2d_list, row_labels=None, col_labels=None, v_buff=0.8, h_buff=1.3,
      include_outer_lines=False, element_to_mobject=Paragraph)

# Specialized variants:
IntegerTable(table, ...)          # element_to_mobject=Integer
DecimalTable(table, ...)          # element_to_mobject=DecimalNumber
MathTable(table, ...)             # element_to_mobject=MathTex
MobjectTable(table, ...)          # items must be VMobjects
```

**Key Table methods** (cell indexing is **1-based**, `(1,1)` = top-left):
```python
table.get_cell((row, col))                    # returns Polygon outline
table.get_highlighted_cell((row, col), color)  # returns BackgroundRectangle
table.add_highlighted_cell((row, col), color)  # adds highlight behind cell
table.get_entries()                            # VGroup of all entries (flat)
table.get_entries((row, col))                  # single entry mobject
table.get_rows()                               # VGroup of VGroups
table.get_columns()
table.set_column_colors(RED, BLUE, ...)
table.create(lag_ratio=1)                      # AnimationGroup for creation
```

**Gotcha**: `get_cell()` returns a Polygon outline, not a filled rectangle. For filled highlight, use `get_highlighted_cell()`. No built-in `remove_highlighted_cell()` — manage BackgroundRectangle objects manually.

### Custom pixel grid (more flexible than Table)

```python
class PixelGrid(VGroup):
    def __init__(self, values_2d, cell_size=0.6, buff=0, **kwargs):
        super().__init__(**kwargs)
        self.rows = len(values_2d)
        self.cols = len(values_2d[0])
        self.cells = {}   # (r, c) -> Square
        self.labels = {}  # (r, c) -> Text

        for r in range(self.rows):
            for c in range(self.cols):
                sq = Square(side_length=cell_size)
                sq.set_stroke(WHITE, 1)
                sq.set_fill(BLACK, opacity=0.8)
                txt = Text(str(values_2d[r][c]), font_size=20)
                self.cells[(r, c)] = sq
                self.labels[(r, c)] = txt
                self.add(sq, txt)

        VGroup(*[self.cells[(r,c)] for r in range(self.rows)
                 for c in range(self.cols)]).arrange_in_grid(
            rows=self.rows, cols=self.cols, buff=buff
        )
        for (r, c), txt in self.labels.items():
            txt.move_to(self.cells[(r, c)].get_center())

    def highlight_cell(self, r, c, color=YELLOW, opacity=0.5):
        self.cells[(r, c)].set_fill(color, opacity=opacity)

    def update_value(self, r, c, new_val):
        new_txt = Text(str(new_val), font_size=20)
        new_txt.move_to(self.cells[(r, c)].get_center())
        return Transform(self.labels[(r, c)], new_txt)
```

### Sliding window / convolution kernel animation

```python
ROWS, COLS, KERNEL = 5, 5, 3
cell_size = 0.7
cells = VGroup(*[Square(side_length=cell_size, stroke_width=1)
                 for _ in range(ROWS * COLS)])
cells.arrange_in_grid(rows=ROWS, cols=COLS, buff=0)

def get_window_rect(top_r, top_c):
    tl = cells[top_r * COLS + top_c]
    br = cells[(top_r + KERNEL - 1) * COLS + (top_c + KERNEL - 1)]
    return SurroundingRectangle(VGroup(tl, br), color=YELLOW, buff=0.02, stroke_width=3)

window = get_window_rect(0, 0)
self.play(Create(window))

for r in range(ROWS - KERNEL + 1):
    for c in range(COLS - KERNEL + 1):
        if r == 0 and c == 0:
            continue
        new_window = get_window_rect(r, c)
        self.play(Transform(window, new_window), run_time=0.5)
```

### Cell highlighting patterns
```python
# Method 1: animate fill
cell.animate.set_fill(YELLOW, opacity=0.5)

# Method 2: SurroundingRectangle (move between cells)
highlight = SurroundingRectangle(cell, color=YELLOW, buff=0.02)
new_hl = SurroundingRectangle(other_cell, color=YELLOW, buff=0.02)
self.play(ReplacementTransform(highlight, new_hl))

# Method 3: Indicate (temporary flash)
self.play(Indicate(cell, color=RED, scale_factor=1.2))

# Method 4: Dynamic highlight with ValueTracker
row_tracker, col_tracker = ValueTracker(0), ValueTracker(0)
highlight = Square(side_length=cell_size, color=YELLOW, fill_opacity=0.3)
highlight.add_updater(lambda m: m.move_to(
    cells[int(row_tracker.get_value()) * COLS + int(col_tracker.get_value())].get_center()
))
```

### Updating grid values
```python
old_label = labels[(r, c)]
new_label = Text("42", font_size=20).move_to(old_label.get_center())
self.play(Transform(old_label, new_label))
```

### Arrow and label tracking algorithm state
```python
pointer = Arrow(ORIGIN, DOWN * 0.5, buff=0, color=RED)
label = Text("current", font_size=18, color=RED)
label.always.next_to(pointer, UP)

pointer.add_updater(lambda mob: mob.next_to(
    cells[int(row_tracker.get_value()) * COLS + int(col_tracker.get_value())].get_top(),
    UP, buff=0.05
))
self.add(pointer, label)
```

### External libraries for algorithm visualization
- **manim-dsa** (`pip install manim-dsa`): pre-built `MArray`, `MStack`, `MGraph` with styled animations
- **ManimML** (`pip install manim_ml`): CNN convolution visualization, forward pass animations

---

## Medical imaging and data pipeline visualizations

### ImageMobject — loading and displaying images

```python
ImageMobject(filename_or_array, scale_to_resolution=1080, invert=False, image_mode='RGBA')
```

**From NumPy array (direct):**
```python
mri_data = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
img = ImageMobject(mri_data)
img.height = 5  # sizing
self.add(img)
```

**Explicit grayscale→RGBA conversion (recommended for full control):**
```python
def grayscale_to_rgba(gray_array):
    if gray_array.dtype != np.uint8:
        gray_array = np.clip(gray_array * 255, 0, 255).astype(np.uint8)
    h, w = gray_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray_array  # R
    rgba[:, :, 1] = gray_array  # G
    rgba[:, :, 2] = gray_array  # B
    rgba[:, :, 3] = 255         # A
    return rgba

img = ImageMobject(grayscale_to_rgba(mri_slice))
```

**From temporary file:**
```python
import tempfile
from PIL import Image

def numpy_to_temp_image(arr):
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(arr).save(tmp.name)
    return tmp.name

img = ImageMobject(numpy_to_temp_image(data))
```

**Key methods:**
```python
img.set_opacity(alpha)                    # 1=opaque, 0=transparent
img.get_pixel_array()                     # returns np.ndarray (RGBA uint8)
img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
# Available: "nearest", "linear", "cubic", "lanczos", "box"
# Use "nearest" for medical images to avoid blurring pixel boundaries
img.height = 5                            # direct property assignment
img.scale(factor)
```

**Critical gotchas:**
- **`ImageMobject` is NOT a `VMobject`** — use `Group` (not `VGroup`) to group with other objects
- Arrays must be `np.uint8` — float arrays won't render correctly without manual conversion
- Manim may filter out pure black pixels as transparent. For medical images with true black, add a small offset (+1) or use RGBA array with explicit `alpha=255`
- File paths: use absolute paths or place in `assets/` directory. Manim searches `config.assets_dir`

### Before/after image comparison

```python
img_noisy = ImageMobject(noisy_data)
img_denoised = ImageMobject(denoised_data)
img_noisy.height = img_denoised.height = 4

# Use Group (NOT VGroup) for ImageMobjects
left = Group(Text("Noisy", font_size=28).next_to(img_noisy, UP), img_noisy)
right = Group(Text("Denoised", font_size=28).next_to(img_denoised, UP), img_denoised)
layout = Group(left, right).arrange(RIGHT, buff=1.0)
self.play(FadeIn(layout))
```

**Cross-fade between images:**
```python
img2.move_to(img1)
self.add(img1)
self.play(FadeOut(img1), FadeIn(img2), run_time=1.5)
```

**Adding borders:**
```python
border = SurroundingRectangle(image, color=GREEN, buff=0.05, stroke_width=3)
self.add(image, border)
```

### Data processing pipeline (flow diagram)

```python
steps = ["Input\nMRI", "Noise\nRemoval", "Segmentation", "Output"]
colors = [BLUE, GREEN, ORANGE, RED]

boxes = VGroup()
for text, color in zip(steps, colors):
    box = RoundedRectangle(corner_radius=0.2, height=1.0, width=2.0,
                           color=color, fill_opacity=0.15, stroke_width=2)
    lbl = Text(text, font_size=20).move_to(box)
    boxes.add(VGroup(box, lbl))

boxes.arrange(RIGHT, buff=1.5)

arrows = VGroup(*[
    Arrow(boxes[i].get_right(), boxes[i+1].get_left(), buff=0.1, stroke_width=2)
    for i in range(len(boxes) - 1)
])

# Animate step by step
for i, box in enumerate(boxes):
    self.play(FadeIn(box, shift=UP * 0.3), run_time=0.5)
    if i < len(arrows):
        self.play(GrowArrow(arrows[i]), run_time=0.3)
```

**Directed graph approach:**
```python
g = DiGraph(
    vertices, edges,
    layout="tree", root_vertex="Input",
    layout_config={"vertex_spacing": (2, 1.5)},
    vertex_config={"radius": 0.4, "fill_opacity": 0.8},
    labels=True  # auto-labels from vertex names
)
self.play(Create(g))
```
Supports layouts: `"tree"`, `"spring"`, `"circular"`, `"kamada_kawai"`, `"planar"`, `"spectral"`.

### Dynamic PSNR/metric counters

**Pattern 1: ValueTracker + DecimalNumber**
```python
psnr_tracker = ValueTracker(20.0)
psnr_label = Text("PSNR: ", font_size=36)
psnr_number = DecimalNumber(psnr_tracker.get_value(), num_decimal_places=2,
                            unit=r"\text{ dB}", font_size=36)
psnr_number.add_updater(lambda d: d.set_value(psnr_tracker.get_value()))
psnr_number.add_updater(lambda d: d.next_to(psnr_label, RIGHT))

VGroup(psnr_label, psnr_number).to_corner(UR)
self.add(psnr_label, psnr_number)
self.play(psnr_tracker.animate.set_value(35.5), run_time=3)
```

**Pattern 2: Variable class (built-in label + value)**
```python
psnr_var = Variable(20.0, Text("PSNR", font_size=28), num_decimal_places=2)
ssim_var = Variable(0.75, Text("SSIM", font_size=28), num_decimal_places=4)
psnr_var.to_corner(UR)
ssim_var.next_to(psnr_var, DOWN)

self.play(Write(psnr_var), Write(ssim_var))
self.play(
    psnr_var.tracker.animate.set_value(35.5),
    ssim_var.tracker.animate.set_value(0.96),
    run_time=3
)
```

**Pattern 3: always_redraw for dynamic text**
```python
psnr_tracker = ValueTracker(20.0)
psnr_text = always_redraw(
    lambda: Text(f"PSNR: {psnr_tracker.get_value():.2f} dB", font_size=30).to_corner(UR)
)
self.add(psnr_text)
self.play(psnr_tracker.animate.set_value(35.5), run_time=3)
```

**DecimalNumber API:**
```python
DecimalNumber(number=0, num_decimal_places=2, mob_class=MathTex, include_sign=False,
              group_with_commas=True, unit=None, edge_to_fix=LEFT, font_size=48)
# Methods: set_value(val), get_value(), increment_value(delta)
# edge_to_fix=LEFT prevents jumping as digit count changes
```

---

## Scientific charts and plots

### Axes class

```python
Axes(x_range=None, y_range=None, x_length=12, y_length=6,
     axis_config=None, x_axis_config=None, y_axis_config=None, tips=True)
# x_range/y_range: (min, max, step) — step determines tick spacing
```

**Key axis_config options:**
```python
axis_config={
    "include_numbers": True,
    "font_size": 24,
    "color": GREEN,
    "include_tip": False,
    "decimal_number_config": {"num_decimal_places": 2},
    "scaling": LogBase(custom_labels=True),  # log scale
}
```

### Coordinate conversion (critical)
```python
point = ax.coords_to_point(x, y)   # or ax.c2p(x, y)
coords = ax.point_to_coords(point)  # or ax.p2c(point)
point = ax.input_to_graph_point(x_val, graph)  # or ax.i2gp(x_val, graph)
```

**Gotcha**: `c2p(1, 2)` is NOT the same as scene point `(1, 2, 0)`. Always use `c2p` for placing objects on axes.

### Plotting continuous functions
```python
graph = ax.plot(function, x_range=None, use_smoothing=True, color=BLUE, stroke_width=3)
# x_range overrides axes range: [x_min, x_max, x_step]
# Decrease x_step for rapidly-changing functions:
graph = ax.plot(func, x_range=[0.001, 10, 0.01], color=GREEN)
```

**`plot()` vs `get_graph()`**: In CE v0.17+, `plot()` is the current API. `get_graph()` is a legacy alias.

**Warning**: Manim samples evenly-spaced points and interpolates with Bézier curves. Default sampling may miss sharp features. Decrease x_step or use `use_smoothing=False` for discontinuous functions.

### Plotting discrete data points
```python
line_graph = ax.plot_line_graph(
    x_values=[0, 1.5, 2, 2.8, 4],
    y_values=[1, 3, 2.25, 4, 2.5],
    line_color=GOLD_E,
    add_vertex_dots=True,
    vertex_dot_radius=0.08,
    vertex_dot_style={"fill_color": PURPLE, "stroke_width": 3},
    stroke_width=4,
)
# Returns dict: line_graph["line_graph"], line_graph["vertex_dots"]
```

**Scatter plot (dots only, no lines):**
```python
scatter = ax.plot_line_graph(
    x_values=xs, y_values=ys,
    add_vertex_dots=True,
    stroke_width=0,  # hide connecting lines = scatter only
    vertex_dot_style={"fill_color": RED},
)
```

**Manual dots:**
```python
dots = VGroup(*[Dot(ax.c2p(x, y), color=BLUE, radius=0.06) for x, y in data])
self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.1))
```

### Other plot types
```python
# Parametric curve
curve = ax.plot_parametric_curve(
    lambda t: np.array([np.cos(t), 2*np.sin(t)]), t_range=(0, 2*PI, 0.1))

# Implicit curve (f(x,y) = 0)
curve = ax.plot_implicit_curve(lambda x, y: x**2 + y**2 - 1, color=YELLOW)

# Derivative / antiderivative
deriv = ax.plot_derivative_graph(graph, color=GREEN)
```

### BarChart class

```python
BarChart(values, bar_names=None, y_range=None, x_length=None, y_length=None,
         bar_colors=['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
         bar_width=0.6, bar_fill_opacity=0.7, bar_stroke_width=3)
# Inherits from Axes — all Axes methods work on BarChart
```

**Animated bar value changes:**
```python
chart = BarChart(values=[28, 0, 0, 0, 0], y_range=[0, 40, 5])
self.play(Create(chart))
self.play(chart.animate.change_bar_values([33, 36, 29, 34, 4]), run_time=3)

# Bar labels
c_bar_lbls = chart.get_bar_labels(font_size=24, buff=0.25)
```

### Area under curve
```python
area = ax.get_area(curve, x_range=(PI/2, 3*PI/2), color=(GREEN_B, GREEN_D), opacity=0.5)
# Between two curves:
area = ax.get_area(curve_2, [2, 3], bounded_graph=curve_1, color=GREY, opacity=0.5)

# Riemann rectangles
rects = ax.get_riemann_rectangles(graph, x_range=[0, 4], dx=0.1,
    input_sample_type='left', color=(BLUE, GREEN), fill_opacity=1)
```

### Axis labels and annotations
```python
labels = ax.get_axis_labels(x_label="x", y_label="f(x)")
ax.add_coordinates()                     # default numbers
ax.add_coordinates(range(-4, 5), None)   # custom x labels

# Graph label
label = ax.get_graph_label(graph, label="\\sin(x)", x_val=PI, direction=UP, dot=True)

# Vertical/horizontal lines to a point
v_line = ax.get_vertical_line(ax.i2gp(x_val, graph), color=YELLOW)
h_line = ax.get_horizontal_line(ax.c2p(x, y), color=BLUE)

# T-label (triangle marker)
t_label = ax.get_T_label(x_val=4, graph=func, label=Tex("x"))
```

### Custom legend (no built-in class)
```python
def create_legend(entries):
    """entries: list of (color, label_text) tuples"""
    legend = VGroup()
    for color, text in entries:
        line = Line(ORIGIN, RIGHT * 0.5, color=color, stroke_width=4)
        label = Text(text, font_size=20).next_to(line, RIGHT, buff=0.15)
        legend.add(VGroup(line, label))
    legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    legend.add_background_rectangle(color=BLACK, opacity=0.7, buff=0.15)
    return legend

legend = create_legend([(BLUE, "Method A"), (RED, "Method B")])
legend.to_corner(UR)
```

### Multi-series plot (PSNR vs noise density)
```python
ax = Axes(x_range=[0, 0.5, 0.1], y_range=[10, 45, 5],
          x_length=8, y_length=5, tips=False,
          axis_config={"include_numbers": True, "font_size": 20})
labels = ax.get_axis_labels(
    x_label=Text("Noise Density", font_size=22),
    y_label=Text("PSNR (dB)", font_size=22))

graph_a = ax.plot_line_graph(noise_x, psnr_a, line_color=BLUE, stroke_width=3)
graph_b = ax.plot_line_graph(noise_x, psnr_b, line_color=RED, stroke_width=3)

self.play(Write(ax), Write(labels))
self.play(Create(graph_a["line_graph"]), Create(graph_b["line_graph"]))
self.play(FadeIn(graph_a["vertex_dots"]), FadeIn(graph_b["vertex_dots"]))
```

### Creation animations for plots

| Animation | Behavior | Best For |
|-----------|----------|----------|
| `Create(mob)` | Draws stroke progressively start→end | Curves, graphs, lines |
| `Write(mob)` | Draws border then fills | Text, axes |
| `DrawBorderThenFill(mob)` | Outline first, then fill | Filled shapes, bars |
| `FadeIn(mob)` | Opacity 0→1 | Any mobject |

**Note**: `ShowCreation` was renamed to `Create` in CE v0.7+. `ShowCreation` is deprecated.

```python
# Recommended pattern for scientific plots:
self.play(Write(ax), Write(labels))       # axes appear
self.play(Create(graph), run_time=2)       # curve draws itself
self.play(FadeIn(area))                    # area fades in
```

### Animating graph transitions
```python
graph1 = ax.plot(lambda x: np.sin(x), color=BLUE)
graph2 = ax.plot(lambda x: np.cos(x), color=RED)
self.play(Create(graph1))
self.play(Transform(graph1, graph2))  # morphs sin → cos
```

### Moving dot along curve
```python
dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
self.play(MoveAlongPath(dot, graph, rate_func=linear, run_time=3))
```

### NumberPlane (axes with grid)
```python
NumberPlane(x_range=(-7, 7, 1), y_range=(-4, 4, 1),
            background_line_style={"stroke_color": TEAL, "stroke_opacity": 0.6})
# Inherits ALL Axes methods. Adds background grid lines.
# Non-linear transforms:
plane.prepare_for_nonlinear_transform()
self.play(plane.animate.apply_function(
    lambda p: p + np.array([np.sin(p[1]), np.sin(p[0]), 0])
))
```

---

## ValueTracker and updaters

### ValueTracker API
```python
tracker = ValueTracker(initial_value)
tracker.get_value()                          # current value
tracker.set_value(5)                         # instant (no animation)
self.play(tracker.animate.set_value(5))      # animated
self.play(tracker.animate.increment_value(2)) # animated increment
```

### Updater patterns

**Pattern 1: `add_updater` — runs every frame**
```python
label.add_updater(lambda m: m.next_to(dot, UP))           # position tracking
mob.add_updater(lambda m, dt: m.rotate(0.1 * dt))          # time-based (dt)
mob.remove_updater(func)                                    # remove specific
mob.clear_updaters()                                        # remove all
```

**Pattern 2: `always_redraw` — reconstruct every frame**
```python
line = always_redraw(lambda: Line(dot1.get_center(), dot2.get_center(), color=RED))
self.add(line)
```

**Pattern 3: `.always` syntax (simple cases)**
```python
text.always.next_to(moving_square, UP)
# NOT compatible with ValueTracker.get_value()
```

**Pattern 4: `.become()` inside updater**
```python
brace.add_updater(lambda b: b.become(Brace(changing_line, DOWN)))
```

**Utility functions:**
```python
always_shift(mob, direction=RIGHT, rate=0.1)   # continuous shift
always_rotate(mob, rate=20*DEGREES)             # continuous rotation
turn_animation_into_updater(anim, cycle=False)  # convert animation to updater
```

**Critical gotchas:**
- **MathTex/Tex cannot be updated in place** — must be recreated. Use `always_redraw(lambda: MathTex(...))` or `mob.become(MathTex(...))`
- **`suspend_mobject_updating=True`** (default in `self.play()`) pauses updaters during that animation. Set to `False` if needed: `self.play(..., suspend_mobject_updating=False)`
- `dt` parameter in updaters is time delta per frame, not cumulative
- Mobjects with updaters must be added to scene via `self.add()` before the play call

### Dot tracking a function with ValueTracker
```python
t = ValueTracker(0)
def func(x): return 2 * (x - 5) ** 2

graph = ax.plot(func, color=MAROON)
dot = Dot(ax.c2p(t.get_value(), func(t.get_value())))
dot.add_updater(lambda d: d.move_to(ax.c2p(t.get_value(), func(t.get_value()))))
self.add(ax, graph, dot)
self.play(t.animate.set_value(5), run_time=3)
```

---

## Custom mobjects

### Subclassing VMobject
```python
class CustomShape(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # CRITICAL: self.add() all sub-mobjects — they won't render otherwise
        self.rect = Rectangle(width=3, height=1.5)
        self.label = Text("Hello", font_size=24).move_to(self.rect)
        self.add(self.rect, self.label)
```

### Subclassing VGroup (simpler composite)
```python
class LabeledBox(VGroup):
    def __init__(self, text, color=BLUE, **kwargs):
        super().__init__(**kwargs)
        self.box = RoundedRectangle(corner_radius=0.2, width=2, height=1, color=color)
        self.label = Text(text, font_size=20).move_to(self.box)
        self.add(self.box, self.label)
```

**Key rules:**
1. Always call `super().__init__(**kwargs)` first
2. Always `self.add()` all sub-mobjects
3. `VMobject`/`VGroup` only accepts other `VMobject` instances
4. `.animate` syntax works automatically on custom subclasses
5. Override `generate_points()` for custom Bézier geometry

---

## Camera and rendering

### MovingCameraScene — 2D pan/zoom
```python
class ZoomExample(MovingCameraScene):
    def construct(self):
        sq = Square(color=BLUE).move_to(2 * LEFT)
        self.add(sq)

        # Zoom into square
        self.play(self.camera.frame.animate.move_to(sq).set(width=sq.width * 2))

        # Save/restore camera state
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.5).move_to(sq))
        self.play(Restore(self.camera.frame))

        # auto_zoom (v0.18+)
        self.play(self.camera.auto_zoom(sq, margin=1))
```

`self.camera.frame` is a Mobject — animate with `.animate`, use `save_state()`/`Restore()`, add updaters.

### ThreeDScene — 3D rendering
```python
class My3D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70*DEGREES, theta=-135*DEGREES)
        axes = ThreeDAxes()
        surface = Surface(lambda u, v: axes.c2p(u, v, np.sin(u)*np.cos(v)),
                          u_range=[-PI, PI], v_range=[-PI, PI])
        self.add(axes, surface)
        self.move_camera(phi=45*DEGREES, theta=-60*DEGREES, run_time=2)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # 2D overlays in 3D scene
        label = Text("Hello").to_corner(UL)
        self.add_fixed_in_frame_mobjects(label)
```

### Frame/background settings
```python
config.frame_width = 16    # logical units (default 14.222)
config.frame_height = 9    # default 8.0
config.background_color = "#1e1e2e"
```

---

## Color, style, and rate functions

### Color system
```python
from manim import ManimColor, color_gradient, interpolate_color

# Accepts: named constants (RED, GREEN_A), hex strings ("#FC6255"), RGB tuples
c_interp = interpolate_color(RED, BLUE, 0.5)  # midpoint
grad = color_gradient([RED, YELLOW, GREEN], 10)  # 10 interpolated colors
```

### VMobject style parameters
```python
mob.set_fill(color=RED, opacity=0.8)
mob.set_stroke(color=WHITE, width=2, opacity=1.0)
mob.set_color(BLUE)                          # sets both fill and stroke
mob.set_opacity(0.5)                         # sets both opacities
mob.set_color_by_gradient(RED, YELLOW, GREEN) # gradient across submobjects
```

**Gotcha**: `color=RED` in constructor sets both fill_color and stroke_color. Use `fill_color` and `stroke_color` separately for independent control.

### Rate functions
All map `t ∈ [0,1] → [0,1]`. Common ones:

| Function | Behavior |
|----------|----------|
| `smooth` | Ease in-out (default for most animations) |
| `linear` | Constant speed |
| `rush_into` | Fast start, smooth stop |
| `rush_from` | Smooth start, fast end |
| `there_and_back` | Goes to 1 at t=0.5, back to 0 |
| `there_and_back_with_pause` | Same but holds at 1 |
| `running_start` | Pulls back before going forward |
| `exponential_decay` | Fast start, exponential slowdown |
| `smoothstep` / `smootherstep` | Higher-order smooth step functions |

Standard easing: `rate_functions.ease_in_sine`, `ease_out_quad`, `ease_in_out_cubic`, etc.

```python
self.play(mob.animate.shift(RIGHT * 3), rate_func=rush_into, run_time=2)
# Per-animation rate functions:
self.play(
    a.animate(rate_func=smooth).shift(LEFT),
    b.animate(rate_func=linear).shift(RIGHT),
)
```

---

## Essential utility patterns

### `.animate` syntax
```python
self.play(mob.animate.shift(RIGHT).set_color(RED).scale(0.5))
self.play(mob.animate(run_time=2, rate_func=there_and_back).rotate(PI / 4))
```
**Gotcha**: `.animate` interpolates linearly between start/end states. `.animate.rotate(2*PI)` produces no visible movement (start=end). Use `Rotate(mob, angle=2*PI)` instead.

### `.copy()` and `.become()`
```python
sq2 = sq.copy()              # deep copy — independent mobject
mob.become(other_mob)         # instant transform (no animation)
```

### `.set_z_index()`
```python
circle.set_z_index(1)
square.set_z_index(2)        # renders on top of circle
# Arrow tip gotcha: tip doesn't inherit z_index
arrow.set_z_index(10)
arrow.get_tip().set_z_index(10)  # must set separately
```

### `save_state()` / `Restore()`
```python
mob.save_state()
self.play(mob.animate.scale(2).set_color(RED))
self.play(Restore(mob))     # back to original
```

### `MoveToTarget`
```python
mob.generate_target()
mob.target.shift(UP).set_color(RED)
self.play(MoveToTarget(mob))
```

### Fade out everything
```python
self.play(*[FadeOut(mob) for mob in self.mobjects])
```

### Method chaining
```python
Circle().set_fill(RED, 0.5).set_stroke(WHITE, 2).shift(UP).scale(0.5)
```

### Professional styling tips
- Use `tips=False` for clean scientific charts
- Use `StealthTip` for sleeker arrow tips: `axis_config={"tip_shape": StealthTip}`
- Convert to dashed: `DashedVMobject(graph, num_dashes=30)`
- Background rect for labels: `label.add_background_rectangle(color=BLACK, opacity=0.7, buff=0.1)`
- z-index for layering: `area.set_z_index(-1)` behind curves

---

## Notable plugins

- **manim-slides** — presentation tool; `Slide` instead of `Scene`, `self.next_slide()` for pauses. Render: `manim-slides render`, present: `manim-slides present`
- **manim-voiceover** — add TTS voiceovers (Azure, Google, OpenAI) or record. Syncs audio duration to animation via `tracker.duration`
- **manim-physics** — rigid body dynamics, electromagnetism, waves
- **manim-revealjs** — embed in Reveal.js slides
- **manim-chemistry** — molecules, orbitals, Bohr diagrams
- **manim-editor** — web-based presentation tool

---

## Version notes (Manim CE v0.17–v0.20+)

- **v0.17**: Stable baseline. Double-brace `{{ }}` syntax. ManimColor system.
- **v0.18.0**: `always_redraw` regression with single ValueTracker (fixed in patches). `LaggedStart` updater bug (#3950).
- **v0.19.x**: `VGroup` accepts iterables (lists, generators) — automatically unpacked.
- **v0.20.x**: Current stable (early 2026). `MovingCamera.auto_zoom()` added. `ChangeSpeed` animation. `Blink` animation.
- OpenGL renderer remains experimental; Cairo is the default and best-documented.
- `ShowCreation` renamed to `Create` in CE v0.7+. Always use `Create`.
- `GREEN_C == GREEN` — `_C` suffix aliases exist for backward compatibility.

---

## Key import reference

```python
from manim import (
    # Core
    Scene, MovingCameraScene, ThreeDScene, Group, VGroup,
    # Images
    ImageMobject, RESAMPLING_ALGORITHMS,
    # Shapes
    Rectangle, RoundedRectangle, Square, Circle, Dot, Line, Arrow, DashedLine,
    # Shape matchers
    SurroundingRectangle, BackgroundRectangle, Brace, BraceLabel,
    # Text
    Text, MarkupText, Tex, MathTex,
    # Numbers
    DecimalNumber, Integer, Variable,
    # Value tracking
    ValueTracker,
    # Tables
    Table, IntegerTable, DecimalTable, MathTable, MobjectTable,
    # Charts & Axes
    Axes, NumberPlane, ThreeDAxes, BarChart,
    # Graphs (network diagrams)
    Graph, DiGraph,
    # Animations - creation
    FadeIn, FadeOut, FadeTransform, Create, Write, DrawBorderThenFill, GrowArrow,
    GrowFromCenter, SpinInFromNothing,
    # Animations - transform
    Transform, ReplacementTransform, TransformFromCopy,
    TransformMatchingTex, TransformMatchingShapes, MoveToTarget, Restore,
    # Animations - indication
    Indicate, Circumscribe, Flash, FocusOn, ApplyWave, Wiggle,
    # Animations - movement
    MoveAlongPath, Rotate,
    # Animation groups
    AnimationGroup, Succession, LaggedStart, LaggedStartMap,
    # Updater helpers
    always_redraw, always_shift, always_rotate,
    # Rate functions
    smooth, linear, rush_into, rush_from, there_and_back,
    there_and_back_with_pause, running_start, exponential_decay,
    # 3D
    Surface, ThreeDAxes,
    # Constants
    UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR,
    PI, TAU, DEGREES,
    SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF, LARGE_BUFF,
    # Colors
    WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ORANGE, GREY, GRAY,
    BLUE_C, GREEN_C, RED_C, TEAL, MAROON, GOLD_E, DARK_GRAY,
    # Color utilities
    ManimColor, color_gradient, interpolate_color,
    # Config
    config, tempconfig, TexTemplate,
)
import numpy as np
```