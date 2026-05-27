from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/sharif/telecom/final-year-project/NCC/ucc-submission")
OUTPUT = ROOT / "assets" / "project-timeline-gantt.png"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    width_chars: int,
    line_height: int,
    fill: str,
    font_obj: ImageFont.FreeTypeFont,
) -> None:
    x, y = xy
    for line in wrap(text, width=width_chars):
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_height


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font_obj: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        (left + (right - left - text_w) / 2, top + (bottom - top - text_h) / 2 - 2),
        text,
        font=font_obj,
        fill=fill,
    )


def main() -> None:
    width, height = 2200, 900
    margin = 34
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    teal = "#007367"
    teal_dark = "#005E56"
    grid = "#B7DFF1"
    header_fill = "#F2FBFE"
    text = "#061728"

    header_font = font(BOLD, 23)
    small_header_font = font(BOLD, 20)
    phase_font = font(REGULAR, 22)

    table_left = margin
    table_top = margin
    phase_col_w = 520
    week_w = 58
    month_h = 62
    week_h = 58
    row_h = 108
    months = ["May", "June", "July", "August", "September", "October", "November"]
    weeks_per_month = 4
    total_weeks = len(months) * weeks_per_month
    grid_w = week_w * total_weeks
    table_w = phase_col_w + grid_w
    table_h = month_h + week_h + row_h * 6

    draw.rectangle((table_left, table_top, table_left + table_w, table_top + table_h), fill="#FFFFFF")
    draw.rectangle((table_left, table_top, table_left + table_w, table_top + table_h), outline=grid, width=3)

    # Header bands
    draw.rectangle((table_left, table_top, table_left + phase_col_w, table_top + month_h + week_h), fill=header_fill)
    center_text(
        draw,
        (table_left, table_top, table_left + phase_col_w, table_top + month_h + week_h),
        "Phase",
        font_obj=header_font,
        fill=text,
    )

    grid_left = table_left + phase_col_w
    for idx, month in enumerate(months):
        x0 = grid_left + idx * weeks_per_month * week_w
        x1 = x0 + weeks_per_month * week_w
        draw.rectangle((x0, table_top, x1, table_top + month_h), fill=header_fill, outline=grid, width=2)
        center_text(draw, (x0, table_top, x1, table_top + month_h), month, font_obj=header_font, fill=text)

    week_y = table_top + month_h
    for week in range(1, total_weeks + 1):
        x0 = grid_left + (week - 1) * week_w
        x1 = x0 + week_w
        draw.rectangle((x0, week_y, x1, week_y + week_h), fill="#F2FCFE", outline=grid, width=2)
        center_text(draw, (x0, week_y, x1, week_y + week_h), str(week), font_obj=small_header_font, fill=text)

    phases = [
        ("Project initiation, scope, approvals", 1, 4),
        ("Dataset preparation and impairment modelling", 3, 8),
        ("DAE design, training, reconstruction checks", 7, 13),
        ("AMC baseline and integrated pipeline testing", 11, 18),
        ("Android-SDR prototype workflow and field prep", 16, 22),
        ("Final analysis, validation, reporting, dissemination", 21, 28),
    ]

    body_top = table_top + month_h + week_h
    for row_idx, (label, start, end) in enumerate(phases):
        y0 = body_top + row_idx * row_h
        y1 = y0 + row_h
        fill = "#FFFFFF" if row_idx % 2 == 0 else "#F8FDFF"
        draw.rectangle((table_left, y0, table_left + table_w, y1), fill=fill, outline=grid, width=2)
        draw.rectangle((table_left, y0, table_left + phase_col_w, y1), fill=fill, outline=grid, width=2)
        draw_wrapped(
            draw,
            label,
            (table_left + 16, y0 + 24),
            width_chars=38,
            line_height=28,
            fill=text,
            font_obj=phase_font,
        )

        for week in range(total_weeks + 1):
            x = grid_left + week * week_w
            draw.line((x, y0, x, y1), fill=grid, width=2)

        bar_x0 = grid_left + (start - 1) * week_w + 2
        bar_x1 = grid_left + end * week_w - 2
        bar_y0 = y0 + 9
        bar_y1 = y1 - 9
        draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), fill=teal, outline=teal_dark, width=2)

    # Stronger monthly separators across body.
    for month_idx in range(len(months) + 1):
        x = grid_left + month_idx * weeks_per_month * week_w
        draw.line((x, table_top, x, table_top + table_h), fill="#8FCBE6", width=3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    crop_box = (
        table_left,
        table_top,
        table_left + table_w + 3,
        table_top + table_h + 3,
    )
    img.crop(crop_box).save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
