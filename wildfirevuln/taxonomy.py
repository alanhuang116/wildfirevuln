"""Harmonised building taxonomy for DINS records.

DINS changed its coding over time. 2013-2017 records roof and siding as
'Combustible' / 'Fire Resistant'; 2018 records siding as 'Combustible' /
'Ignition Resistant'; from 2019 siding and roof are recorded by material.
Everything here maps onto one taxonomy. 'unknown' collects Unknown, blank,
NA and codes that belong to a different field (data-entry spill-over).
"""
from __future__ import annotations

UNKNOWN = "unknown"

DAMAGE = {
    "No Damage": 0,
    "Affected (>0-10%)": 1,
    "Minor (10-25%)": 2,
    "Major (25-50%)": 3,
    "Destroyed (>50%)": 4,
}
# Damage ratio assigned to each DINS band when converting to expected loss.
# Destroyed is treated as a total loss of the structure, which is how
# California carriers settle >50% fire damage in practice.
DAMAGE_RATIO = {0: 0.0, 1: 0.05, 2: 0.175, 3: 0.375, 4: 1.0}

RESIDENTIAL = {"Single Residence", "Multiple Residence"}
# Commercial real estate and other fixed assets. DINS splits commercial
# buildings only by storey count plus a few institutional types; it has no
# warehouse, industrial, office or retail code.
COMMERCIAL = {"Nonresidential Commercial", "Mixed Commercial/Residential",
              "Infrastructure"}
COMMERCIAL_STRUCT = ["com_1", "com_2", "institutional", "mixed", "infrastructure"]

STRUCT = {
    "Single Family Residence Single Story": "sfr_1",
    "Single Famliy Residence Single Story": "sfr_1",
    "Single Family Residence Multi Story": "sfr_2",
    "Mobile Home Single Wide": "mobile",
    "Mobile Home Double Wide": "mobile",
    "Mobile Home Triple Wide": "mobile",
    "Motor Home": "motorhome",
    "Motor Home/Travel Trailer": "motorhome",
    "Multi Family Residence Single Story": "multi",
    "Multi Family Residence Multi Story": "multi",
    "Mixed Commercial/Residential": "mixed",
    "Commercial Building Single Story": "com_1",
    "Commercial Building Multi Story": "com_2",
    "School": "institutional", "Church": "institutional", "Hospital": "institutional",
    "Infrastructure": "infrastructure", "Utility Misc Structure": "infrastructure",
}

ROOF = {"Tile": "tile", "Concrete": "tile", "Concrete Slab": "tile",
        "Metal": "metal", "Asphalt": "asphalt", "Wood": "wood",
        "Combustible": "wood"}
EAVES = {"Enclosed": "enclosed", "Unenclosed": "open", "No Eaves": "none"}
VENTS = {'Mesh Screen <= 1/8"': "fine", 'Mesh Screen > 1/8"': "coarse",
         "Unscreened": "open", "No Vents": "none"}
SIDING = {"Wood": "combustible", "Vinyl": "combustible",
          "Combustible": "combustible",
          "Stucco Brick Cement": "noncomb", "Stucco/Brick/Cement": "noncomb",
          "Metal": "noncomb", "Ignition Resistant": "noncomb",
          "Fire Resistant": "noncomb"}
WINDOWS = {"Multi Pane": "multi", "Single Pane": "single"}
DECK = {"Wood": "combustible", "Composite": "combustible",
        "Masonry/Concrete": "none", "No Deck/Porch": "none"}
ATTACH = {"Combustible": "combustible", "Non Combustible": "noncomb",
          "No Patio Cover/Carport": "none", "No Fence": "none"}

# field -> (DINS column, map, reference level, display name)
FIELDS = {
    "roof": ("ROOFCONSTRUCTION", ROOF, "asphalt", "Roof covering"),
    "eaves": ("EAVES", EAVES, "open", "Eaves"),
    "vents": ("VENTSCREEN", VENTS, "coarse", "Vent screening"),
    "siding": ("EXTERIORSIDING", SIDING, "combustible", "Exterior siding"),
    "windows": ("WINDOWPANE", WINDOWS, "single", "Window glazing"),
    "deck": ("DECKPORCHELEVATED", DECK, "combustible", "Elevated deck"),
    "patio": ("PATIOCOVERCARPORT", ATTACH, "combustible", "Attached patio cover"),
    "fence": ("FENCEATTACHEDTOSTRUCTURE", ATTACH, "combustible", "Attached fence"),
}

LEVEL_NAMES = {
    "roof": {"tile": "Tile / concrete", "metal": "Metal", "asphalt": "Asphalt shingle",
             "wood": "Wood shake / combustible"},
    "eaves": {"enclosed": "Enclosed (boxed)", "open": "Open / unenclosed", "none": "No eaves"},
    "vents": {"fine": "Mesh ≤ 1/8″", "coarse": "Mesh > 1/8″",
              "open": "Unscreened", "none": "No vents"},
    "siding": {"combustible": "Wood / vinyl", "noncomb": "Stucco, brick, cement, metal"},
    "windows": {"multi": "Multi-pane", "single": "Single-pane"},
    "deck": {"combustible": "Wood / composite", "none": "None or masonry"},
    "patio": {"combustible": "Combustible", "noncomb": "Non-combustible", "none": "None"},
    "fence": {"combustible": "Combustible", "noncomb": "Non-combustible", "none": "None"},
    "struct": {"sfr_1": "Single-family, one storey", "sfr_2": "Single-family, multi-storey",
               "mobile": "Manufactured / mobile home", "motorhome": "Motor home / trailer",
               "multi": "Multi-family",
               "com_1": "Commercial, one storey", "com_2": "Commercial, multi-storey",
               "institutional": "School, church, hospital", "mixed": "Mixed commercial / residential",
               "infrastructure": "Infrastructure, utility"},
    "era": {"pre1990": "Before 1990", "1990_2007": "1990–2007",
            "post2008": "2008 or later (Chapter 7A)"},
}


# Bank collateral types mapped to the nearest DINS class. 'proxy' means DINS
# has no code for the type and the mapping is by construction and footprint.
CRE_MAP = {
    "warehouse": ("com_1", "proxy"), "industrial": ("com_1", "proxy"),
    "logistics": ("com_1", "proxy"), "retail": ("com_1", "proxy"),
    "office": ("com_2", "proxy"), "hotel": ("com_2", "proxy"),
    "commercial_1": ("com_1", "direct"), "commercial_2": ("com_2", "direct"),
    "school": ("institutional", "direct"), "church": ("institutional", "direct"),
    "hospital": ("institutional", "direct"), "mixed_use": ("mixed", "direct"),
    "multifamily": ("multi", "direct"), "utility": ("infrastructure", "direct"),
}


def harmonise(value, mapping):
    if not isinstance(value, str):
        return UNKNOWN
    return mapping.get(value.strip(), UNKNOWN)


def era(year):
    try:
        y = int(year)
    except (TypeError, ValueError):
        return UNKNOWN
    if y < 1800:
        return UNKNOWN
    if y < 1990:
        return "pre1990"
    if y < 2008:
        return "1990_2007"
    return "post2008"
