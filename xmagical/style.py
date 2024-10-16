"""Common visual settings and color tools for this benchmark."""

import colorsys
from typing import Tuple

RGBTuple = Tuple[float, float, float]


# Global variable to store the environment
current_env = "xmagical"  # default environment

def set_environment(env: str):
    """
    Set the current environment for the style configuration.
    
    Args:
        env (str): The environment to set, e.g., "magical" or any other.
    """
    global current_env
    current_env = env

def get_environment():
    return current_env

def rgb(r: float, g: float, b: float) -> RGBTuple:
    return (r / 255.0, g / 255.0, b / 255.0)


def darken_rgb(rgb: RGBTuple) -> RGBTuple:
    """Produce a darker version of a base color."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hls_new = (h, max(0, l * 0.9), s)
    return colorsys.hls_to_rgb(*hls_new)


def lighten_rgb(rgb: RGBTuple, times: float = 1.0) -> RGBTuple:
    """Produce a lighter version of a given base color."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    mult = 1.4 ** times
    hls_new = (h, 1 - (1 - l) / mult, s)
    return colorsys.hls_to_rgb(*hls_new)


GOAL_LINE_THICKNESS = 0.01
SHAPE_LINE_THICKNESS = 0.015
ROBOT_LINE_THICKNESS = 0.01
COLORS_RGB_XMAGICAL = {
    # I'm using Berkeley-branded versions of RGBY from
    # https://brand.berkeley.edu/colors/ (lightened).
    "blue": lighten_rgb(rgb(0x3B, 0x7E, 0xA1), 1.7),  # founder's rock
    "yellow": lighten_rgb(rgb(0xFD, 0xB5, 0x15), 1.7),  # california gold
    "red": lighten_rgb(rgb(0xEE, 0x1F, 0x60), 1.7),  # rose garden
    "green": lighten_rgb(rgb(0x85, 0x94, 0x38), 1.7),  # soybean
    "grey": rgb(162, 163, 175),  # cool grey (not sure which one)
    "brown": rgb(224, 171, 118),  # buff
}
COLORS_RGB_MAGICAL = {
    # I'm using Berkeley-branded versions of RGBY from
    # https://brand.berkeley.edu/colors/ (lightened).
    "blue": rgb(0xd4, 0xb4, 0x7b),
    "yellow": rgb(0x83, 0xd5, 0xff),
    "red": lighten_rgb(rgb(0xEE, 0x1F, 0x60), 1.7),  # rose garden
    "green": rgb(0x7d, 0xc8, 0xb2),
    "grey": rgb(0xac, 0xac, 0xaa),
    "brown": rgb(224, 171, 118),  # buff
}

COLORS_RGB = COLORS_RGB_XMAGICAL if current_env == "xmagical" else COLORS_RGB_MAGICAL

# "zoom out" factor when rendering arena; values above 1 will show parts of the
# arena border in allocentric view.
ARENA_ZOOM_OUT = 1.02
