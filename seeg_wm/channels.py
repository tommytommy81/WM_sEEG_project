"""sEEG contact naming, shaft grouping, re-referencing montages and anatomy."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

# <shaft><number>, shaft = letters with optional prime(s) (left hemisphere), e.g. HC'3, PFG'10, OR2
CONTACT_RE = re.compile(r"^(?P<shaft>[A-Za-z]+[A-Za-z0-9]*?['`’]*)(?P<num>\d+)$")


def parse_contact(name: str) -> tuple[str, int] | None:
    """Split a contact name into (shaft, number); ``None`` if it does not parse."""
    m = CONTACT_RE.match(name.strip())
    if m is None:
        return None
    shaft = m.group("shaft").replace("`", "'").replace("’", "'")
    return shaft, int(m.group("num"))


def clean_name(name: str, patterns: list[str]) -> str:
    for pat in patterns or []:
        name = re.sub(pat, "", name)
    return name.replace("`", "'").replace("’", "'")


def hemisphere(name: str) -> str:
    """Prime (') in the shaft name = Left (French/Talairach sEEG convention)."""
    contact = name.split("-")[0]
    parsed = parse_contact(contact)
    shaft = parsed[0] if parsed else contact
    return "Left" if "'" in shaft else "Right"


def group_by_shaft(ch_names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """``{shaft: [(number, name), ...]}`` sorted by contact number."""
    shafts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ch in ch_names:
        parsed = parse_contact(ch)
        if parsed is not None:
            shafts[parsed[0]].append((parsed[1], ch))
    return {k: sorted(v) for k, v in shafts.items()}


def bipolar_pairs(ch_names: list[str], bads: list[str] | None = None,
                  max_gap: int = 1) -> tuple[list[str], list[str]]:
    """Consecutive-contact bipolar pairs along each shaft, skipping bad contacts.

    With ``max_gap=1`` only physically adjacent contacts are paired (HC1-HC2).
    With ``max_gap=2`` a single bad contact may be bridged (HC1-HC3 if HC2 is bad).
    """
    bads = set(bads or [])
    anodes, cathodes = [], []
    for shaft, contacts in group_by_shaft(ch_names).items():
        good = [(n, c) for n, c in contacts if c not in bads]
        for (n1, c1), (n2, c2) in zip(good[:-1], good[1:]):
            if 0 < n2 - n1 <= max_gap:
                anodes.append(c1)
                cathodes.append(c2)
    return anodes, cathodes


def laplacian_neighbors(ch_names: list[str], bads: list[str] | None = None
                        ) -> dict[str, list[str]]:
    """For each good contact, its good immediate neighbours on the same shaft."""
    bads = set(bads or [])
    out = {}
    for shaft, contacts in group_by_shaft(ch_names).items():
        good = {n: c for n, c in contacts if c not in bads}
        for n, c in good.items():
            neigh = [good[k] for k in (n - 1, n + 1) if k in good]
            if neigh:
                out[c] = neigh
    return out


def neighbor_correlation(data: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Median |Pearson r| of each channel with its adjacent contacts on the same shaft.

    Channels without neighbours get NaN.
    """
    idx = {c: i for i, c in enumerate(ch_names)}
    corr = np.full(len(ch_names), np.nan)
    neigh = laplacian_neighbors(ch_names)
    for c, nn in neigh.items():
        x = data[idx[c]]
        rs = [abs(np.corrcoef(x, data[idx[n]])[0, 1]) for n in nn]
        corr[idx[c]] = np.nanmedian(rs)
    return corr


# --------------------------------------------------------------------- anatomy
def assign_regions(ch_names: list[str], region_map: dict[str, list[str]] | None,
                   unassigned: str = "Unassigned") -> dict[str, str]:
    """Map channel names (bipolar or monopolar) to region labels.

    See the header of ``config/regions.yaml`` for the matching rules.
    """
    region_map = region_map or {}
    exact: dict[str, str] = {}
    for region, chans in region_map.items():
        for ch in chans:
            exact.setdefault(ch, region)
    contact_lookup: dict[str, str] = {}
    for ch, region in exact.items():
        for part in ch.split("-"):
            contact_lookup.setdefault(part, region)

    out = {}
    for ch in ch_names:
        if ch in exact:
            out[ch] = exact[ch]
            continue
        region = None
        for part in ch.split("-"):
            if part in exact:          # monopolar entry
                region = exact[part]
                break
        if region is None:
            for part in ch.split("-"):
                if part in contact_lookup:
                    region = contact_lookup[part]
                    break
        out[ch] = region or unassigned
    return out


def merge_hemisphere_label(region: str) -> str:
    return re.sub(r"^(Left|Right|L|R)[\s_-]+", "", region)
