'''
map_visualization.py

Combines CSP map-coloring logic (backtracking, forward checking, singleton
propagation, MRV/Degree/LCV heuristics) with geographic map rendering via
GeoPandas / Matplotlib.

Running this file directly will:
  1. Compute the chromatic number for the USA and Australia maps.
  2. Run 5-trial experiments for all six algorithm/heuristic combinations.
  3. Render colored GeoPandas maps for both regions.
'''

import time
import random

import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Map definitions
# ─────────────────────────────────────────────

# USA: 48 contiguous states + DC  (territories omitted)
USA_NEIGHBORS = {
    "AL": ["FL","GA","MS","TN"],
    "AZ": ["CA","CO","NM","NV","UT"],
    "AR": ["LA","MO","MS","OK","TN","TX"],
    "CA": ["AZ","NV","OR"],
    "CO": ["AZ","KS","NE","NM","OK","UT","WY"],
    "CT": ["MA","NY","RI"],
    "DE": ["MD","NJ","PA"],
    "DC": ["MD","VA"],
    "FL": ["AL","GA"],
    "GA": ["AL","FL","NC","SC","TN"],
    "ID": ["MT","NV","OR","UT","WA","WY"],
    "IL": ["IN","IA","KY","MO","WI"],
    "IN": ["IL","KY","MI","OH"],
    "IA": ["IL","MN","MO","NE","SD","WI"],
    "KS": ["CO","MO","NE","OK"],
    "KY": ["IL","IN","MO","OH","TN","VA","WV"],
    "LA": ["AR","MS","TX"],
    "ME": ["NH"],
    "MD": ["DC","DE","PA","VA","WV"],
    "MA": ["CT","NH","NY","RI","VT"],
    "MI": ["IN","OH","WI"],
    "MN": ["IA","ND","SD","WI"],
    "MS": ["AL","AR","LA","TN"],
    "MO": ["AR","IL","IA","KS","KY","NE","OK","TN"],
    "MT": ["ID","ND","SD","WY"],
    "NE": ["CO","IA","KS","MO","SD","WY"],
    "NV": ["AZ","CA","ID","OR","UT"],
    "NH": ["MA","ME","VT"],
    "NJ": ["DE","NY","PA"],
    "NM": ["AZ","CO","OK","TX"],
    "NY": ["CT","MA","NJ","PA","VT"],
    "NC": ["GA","SC","TN","VA"],
    "ND": ["MN","MT","SD"],
    "OH": ["IN","KY","MI","PA","WV"],
    "OK": ["AR","CO","KS","MO","NM","TX"],
    "OR": ["CA","ID","NV","WA"],
    "PA": ["DE","MD","NJ","NY","OH","WV"],
    "RI": ["CT","MA"],
    "SC": ["GA","NC"],
    "SD": ["IA","MN","MT","ND","NE","WY"],
    "TN": ["AL","AR","GA","KY","MO","MS","NC","VA"],
    "TX": ["AR","LA","NM","OK"],
    "UT": ["AZ","CO","ID","NV","NM","WY"],
    "VT": ["MA","NH","NY"],
    "VA": ["DC","KY","MD","NC","TN","WV"],
    "WA": ["ID","OR"],
    "WV": ["KY","MD","OH","PA","VA"],
    "WI": ["IL","IA","MI","MN"],
    "WY": ["CO","ID","MT","NE","SD","UT"],
}

# Australia: 7 states/territories (ACT omitted per project scope)
AU_NEIGHBORS = {
    "WA":  ["NT","SA"],
    "NT":  ["WA","SA","QLD"],
    "SA":  ["WA","NT","QLD","NSW","VIC"],
    "QLD": ["NT","SA","NSW"],
    "NSW": ["SA","QLD","VIC"],
    "VIC": ["SA","NSW","TAS"],
    "TAS": ["VIC"],
}

# Maps color-number strings ("1", "2", …) to real display colors
COLOR_PALETTE = {
    "1": "#E63946",   # red
    "2": "#457B9D",   # blue
    "3": "#2A9D8F",   # teal
    "4": "#E9C46A",   # yellow
    "5": "#F4A261",   # orange
}

AU_ABBR_TO_NAME = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA":  "South Australia",
    "WA":  "Western Australia",
    "TAS": "Tasmania",
    "NT":  "Northern Territory",
}


# ─────────────────────────────────────────────
#  GeoPandas visualization helpers
# ─────────────────────────────────────────────

def translate_geometries(df, x, y, scale, rotate):
    '''
    Translate, scale, and rotate geometries — used to reposition
    Alaska and Hawaii onto the contiguous US frame.

    :param df: GeoDataFrame to transform
    :param x: x-axis translation offset
    :param y: y-axis translation offset
    :param scale: uniform scale factor
    :param rotate: rotation angle in degrees
    '''
    df.loc[:, "geometry"] = df.geometry.translate(yoff=y, xoff=x)
    center = df.dissolve().centroid.iloc[0]
    df.loc[:, "geometry"] = df.geometry.scale(xfact=scale, yfact=scale, origin=center)
    df.loc[:, "geometry"] = df.geometry.rotate(rotate, origin=center)
    return df


def adjust_us_map(df):
    '''
    Reposition Alaska and Hawaii into inset positions below the
    contiguous states so the full map fits in one frame.

    :param df: GeoDataFrame for the full US
    '''
    df_main   = df[~df.STATEFP.isin(["02", "15"])]
    df_alaska = df[df.STATEFP == "02"]
    df_hawaii = df[df.STATEFP == "15"]

    df_alaska = translate_geometries(df_alaska, 1300000, -4900000, 0.5, 32)
    df_hawaii = translate_geometries(df_hawaii, 5400000, -1500000, 1,   24)

    return pd.concat([df_main, df_alaska, df_hawaii])


def plot_map(region="us", coloring=None):
    '''
    Render a colored geographic map using GeoPandas.

    :param region: "us" or "au"
    :param coloring: dict mapping region abbreviation → hex color string,
                     e.g. {"CA": "#E63946", "TX": "#457B9D"}
                     Unmapped regions are shown in light grey.
    '''
    edge_color       = "#30011E"
    background_color = "#fafafa"

    sns.set_style({
        "font.family": "serif",
        "figure.facecolor": background_color,
        "axes.facecolor":   background_color,
    })

    if region == "us":
        gdf = gpd.read_file("./map_data/us/")
        gdf = gdf[~gdf.STATEFP.isin(["72", "69", "60", "66", "78"])]
        gdf = gdf.to_crs("ESRI:102003")
        gdf = adjust_us_map(gdf)
        key_col = "STUSPS"          # e.g. CA, TX, NY

    elif region == "au":
        gdf = gpd.read_file("./map_data/au/")
        gdf = gdf.to_crs("EPSG:3577")
        key_col = "STE_NAME21"

        # Remove non-mainland entries
        gdf = gdf[~gdf[key_col].isin(
            ["Other Territories", "Outside Australia", "Australian Capital Territory"]
        )]

        # Convert abbreviation keys → full state names
        if coloring:
            coloring = {AU_ABBR_TO_NAME.get(k, k): v for k, v in coloring.items()}

        print(set(gdf[key_col]))

    else:
        raise ValueError("region must be 'us' or 'au'")

    # Apply colors; fall back to grey for any uncolored region
    if coloring:
        gdf["plot_color"] = gdf[key_col].map(coloring).fillna("#dddddd")
    else:
        gdf["plot_color"] = "#dddddd"

    ax = gdf.plot(
        edgecolor=edge_color,
        color=gdf["plot_color"],
        linewidth=1,
    )

    plt.axis("off")
    plt.title(f"{region.upper()} Map Coloring", fontsize=14)
    plt.show()


def assignment_to_hex_coloring(assignment):
    '''
    Convert a CSP assignment dict (region → color-number string) into a
    dict suitable for plot_map (region → hex color string).

    :param assignment: dict[str, str], e.g. {"CA": "1", "NV": "2", …}
    '''
    return {region: COLOR_PALETTE.get(color, "#dddddd")
            for region, color in assignment.items()}


# ─────────────────────────────────────────────
#  CSP Problem class
# ─────────────────────────────────────────────

class MapColoringProblem:
    '''
    Encapsulates a map-coloring CSP.

    Attributes
    ----------
    variables : list[str]
        Ordered list of regions (vertices).
    neighbors : dict[str, list[str]]
        Adjacency list for the map.
    colors : list[str]
        Available color labels ("1", "2", …).
    domains : dict[str, list[str]]
        Current domain for each variable.
    backtracks : int
        Counter incremented on every backtrack step.
    '''

    def __init__(self, neighbors, num_colors):
        self.variables  = list(neighbors.keys())
        self.neighbors  = neighbors
        self.colors     = [str(c) for c in range(1, num_colors + 1)]
        self.domains    = {v: list(self.colors) for v in self.variables}
        self.backtracks = 0

    # ── constraint check ──────────────────────
    def is_consistent(self, var, color, assignment):
        '''
        Return True if assigning *color* to *var* does not conflict with
        any already-assigned neighbor.

        :param var: variable being assigned
        :param color: candidate color
        :param assignment: dict of already-assigned variables
        '''
        for neighbor in self.neighbors[var]:
            if assignment.get(neighbor) == color:
                return False
        return True

    # ── forward checking ──────────────────────
    def forward_check(self, var, color, domains):
        '''
        Remove *color* from every unassigned neighbor's domain.
        Returns False if any domain becomes empty (dead end).

        :param var: variable just assigned
        :param color: color just assigned to var
        :param domains: mutable copy of current domains
        '''
        for neighbor in self.neighbors[var]:
            if color in domains[neighbor]:
                domains[neighbor].remove(color)
                if not domains[neighbor]:
                    return False    # domain wipe-out
        return True

    # ── singleton propagation (AC-1 style) ────
    def propagate_singletons(self, domains):
        '''
        Repeatedly enforce arc-consistency for singleton domains.
        When a variable has only one color left, remove that color from
        all its neighbors.  Continue until stable or a wipe-out occurs.

        Returns False on domain wipe-out, True otherwise.

        :param domains: mutable copy of current domains
        '''
        changed = True
        while changed:
            changed = False
            for var in self.variables:
                if len(domains[var]) == 1:
                    forced_color = domains[var][0]
                    for neighbor in self.neighbors[var]:
                        if forced_color in domains[neighbor]:
                            domains[neighbor].remove(forced_color)
                            changed = True
                            if not domains[neighbor]:
                                return False    # domain wipe-out
        return True

    # ── heuristic helpers ─────────────────────
    def mrv(self, unassigned, domains):
        '''
        Minimum Remaining Values: pick the variable with the fewest
        legal colors remaining in its domain.

        :param unassigned: list of not-yet-assigned variables
        :param domains: current domains dict
        '''
        return min(unassigned, key=lambda v: len(domains[v]))

    def degree(self, unassigned):
        '''
        Degree heuristic: among tied variables, prefer the one with the
        most constraints on remaining unassigned variables.

        :param unassigned: list of not-yet-assigned variables
        '''
        unassigned_set = set(unassigned)
        return max(unassigned,
                   key=lambda v: sum(1 for n in self.neighbors[v]
                                     if n in unassigned_set))

    def lcv(self, var, domains, assignment):
        '''
        Least Constraining Value: order *var*'s colors so that the color
        ruling out the fewest neighbor choices comes first.

        :param var: variable being assigned
        :param domains: current domains dict
        :param assignment: current partial assignment
        '''
        def count_conflicts(color):
            return sum(
                1 for nb in self.neighbors[var]
                if nb not in assignment and color in domains[nb]
            )
        return sorted(domains[var], key=count_conflicts)

    def select_unassigned_variable(self, unassigned, domains, use_heuristics):
        '''
        Choose the next variable to assign.

        With heuristics: MRV first, break ties with Degree.
        Without: return the next variable in the preset order.

        :param unassigned: ordered list of unassigned variables
        :param domains: current domains dict
        :param use_heuristics: bool
        '''
        if not use_heuristics:
            return unassigned[0]

        min_domain     = min(len(domains[v]) for v in unassigned)
        mrv_candidates = [v for v in unassigned if len(domains[v]) == min_domain]

        if len(mrv_candidates) == 1:
            return mrv_candidates[0]

        return self.degree(mrv_candidates)  # tie-break

    def order_domain_values(self, var, domains, assignment, use_heuristics):
        '''
        Return the colors to try for *var*, in order.

        With heuristics: LCV ordering.
        Without: domain as-is.

        :param var: variable being assigned
        :param domains: current domains dict
        :param assignment: current partial assignment
        :param use_heuristics: bool
        '''
        if use_heuristics:
            return self.lcv(var, domains, assignment)
        return list(domains[var])


# ─────────────────────────────────────────────
#  Backtracking search
# ─────────────────────────────────────────────

def backtrack(problem, assignment, unassigned, domains,
              use_fc, use_propagation, use_heuristics):
    '''
    Core recursive backtracking search shared by all six algorithm variants.

    :param problem: MapColoringProblem instance
    :param assignment: dict mapping variable → color (mutated in place)
    :param unassigned: list of variables not yet assigned
    :param domains: dict of current legal colors per variable
    :param use_fc: enable forward checking
    :param use_propagation: enable singleton-domain propagation
    :param use_heuristics: enable MRV / Degree / LCV heuristics
    '''
    if not unassigned:
        return assignment   # complete assignment found

    var       = problem.select_unassigned_variable(unassigned, domains, use_heuristics)
    remaining = [v for v in unassigned if v != var]

    for color in problem.order_domain_values(var, domains, assignment, use_heuristics):
        if problem.is_consistent(var, color, assignment):
            assignment[var] = color

            # Snapshot domains before modification so we can restore on failure
            saved_domains = {v: list(d) for v, d in domains.items()}
            ok = True

            if use_fc:
                ok = problem.forward_check(var, color, domains)

            if ok and use_propagation:
                ok = problem.propagate_singletons(domains)

            if ok:
                result = backtrack(problem, assignment, remaining, domains,
                                   use_fc, use_propagation, use_heuristics)
                if result is not None:
                    return result

            # Restore domains and undo assignment
            for v in domains:
                domains[v] = saved_domains[v]
            del assignment[var]
            problem.backtracks += 1

    return None     # signal backtrack to caller


def solve(neighbors, num_colors, variable_order,
          use_fc, use_propagation, use_heuristics):
    '''
    Set up a MapColoringProblem and run backtracking.

    Returns (assignment | None, backtracks, elapsed_seconds).

    :param neighbors: adjacency dict for the map
    :param num_colors: number of colors available
    :param variable_order: list of variables in the desired trial order
    :param use_fc: enable forward checking
    :param use_propagation: enable singleton propagation
    :param use_heuristics: enable MRV / Degree / LCV heuristics
    '''
    problem          = MapColoringProblem(neighbors, num_colors)
    problem.variables = list(variable_order)
    domains          = {v: list(problem.colors) for v in problem.variables}

    start      = time.perf_counter()
    assignment = backtrack(problem, {}, list(problem.variables), domains,
                           use_fc, use_propagation, use_heuristics)
    elapsed    = time.perf_counter() - start

    return assignment, problem.backtracks, elapsed


# ─────────────────────────────────────────────
#  Chromatic-number helper
# ─────────────────────────────────────────────

def find_chromatic_number(neighbors, max_colors=10):
    '''
    Find the minimum number of colors needed to legally color the graph
    by trying increasing color counts until a solution is found.

    :param neighbors: adjacency dict
    :param max_colors: upper bound before giving up
    '''
    variables = list(neighbors.keys())
    for k in range(1, max_colors + 1):
        assignment, _, _ = solve(neighbors, k, variables,
                                 use_fc=True, use_propagation=True,
                                 use_heuristics=True)
        if assignment is not None:
            return k
    return None


# ─────────────────────────────────────────────
#  Experiment runner and result printer
# ─────────────────────────────────────────────

ALGORITHM_LABELS = ["DFS only", "DFS + FC", "DFS + FC + Prop"]


def run_experiments(neighbors, map_name, num_colors, num_trials=5):
    '''
    Run all six algorithm/heuristic combinations over *num_trials* random
    variable orderings and print a formatted results table.

    :param neighbors: adjacency dict for the map
    :param map_name: human-readable label for printing
    :param num_colors: number of colors to use (should be ≥ chromatic number)
    :param num_trials: number of random orderings to test
    '''
    variables = list(neighbors.keys())

    print(f"\n{'='*72}")
    print(f"  {map_name}  |  {num_colors} colors  |  {num_trials} trials")
    print(f"{'='*72}")

    for section_label, configs in [
        ("WITHOUT Heuristics", [
            (False, False, False),
            (True,  False, False),
            (True,  True,  False),
        ]),
        ("WITH Heuristics (MRV → Degree → LCV)", [
            (False, False, True),
            (True,  False, True),
            (True,  True,  True),
        ]),
    ]:
        print(f"\n── {section_label} ──")
        print(f"{'Trial':<7} | {'Algorithm':<20} | {'Backtracks':>10} | {'Time (s)':>10}")
        print(f"{'-'*7}-+-{'-'*20}-+-{'-'*10}-+-{'-'*10}")

        for trial in range(1, num_trials + 1):
            random.shuffle(variables)
            order = list(variables)

            for label, (fc, prop, heur) in zip(ALGORITHM_LABELS, configs):
                assignment, bt, t = solve(neighbors, num_colors, order, fc, prop, heur)
                status = "✓" if assignment else "✗"
                print(f"{trial:<7} | {label:<20} | {bt:>10} | {t:>10.5f}  {status}")

            print(f"{'-'*7}-+-{'-'*20}-+-{'-'*10}-+-{'-'*10}")


def print_coloring(assignment, map_name):
    '''
    Pretty-print a coloring assignment to the terminal.

    :param assignment: dict mapping region → color-number string
    :param map_name: label shown in the header
    '''
    print(f"\n{map_name} coloring:")
    for region, color in sorted(assignment.items()):
        print(f"  {region:<5} → color {color}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    # ── Chromatic numbers ─────────────────────
    print("Computing chromatic numbers …")
    chi_usa = find_chromatic_number(USA_NEIGHBORS)
    print(f"  USA chromatic number  χ(G) = {chi_usa}")

    chi_au = find_chromatic_number(AU_NEIGHBORS)
    print(f"  Australia chromatic number  χ(G) = {chi_au}")

    # ── Experiments ───────────────────────────
    # chi + 1 keeps every trial tractable while still exercising all variants
    run_experiments(USA_NEIGHBORS, "USA MAP",       chi_usa + 1, num_trials=5)
    run_experiments(AU_NEIGHBORS,  "AUSTRALIA MAP", chi_au,      num_trials=5)

    # ── Solve once for visualization ──────────
    usa_sol, _, _ = solve(USA_NEIGHBORS, chi_usa + 1, list(USA_NEIGHBORS.keys()),
                          use_fc=True, use_propagation=True, use_heuristics=True)
    au_sol,  _, _ = solve(AU_NEIGHBORS,  chi_au,      list(AU_NEIGHBORS.keys()),
                          use_fc=True, use_propagation=True, use_heuristics=True)

    if usa_sol:
        print_coloring(usa_sol, "USA")
        plot_map("us",  assignment_to_hex_coloring(usa_sol))

    if au_sol:
        print_coloring(au_sol, "Australia")
        plot_map("au",  assignment_to_hex_coloring(au_sol))