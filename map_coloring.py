import time, random, sys
from collections import defaultdict

import pandas as pd
import seaborn as sns
import geopandas as gpd

import matplotlib.pyplot as plt


# MAP DEFINITIONS
# USA: dictionary of adjacent regions for the 50 states + DC 
USA_NEIGHBORS = {
    "AK": [],
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
    "HI": [],
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
    "WY": ["CO","ID","MT","NE","SD","UT"]
}

# Australia: dictionary of regions states for the 7 states
AU_NEIGHBORS = {
    "WA":  ["NT","SA"],
    "NT":  ["WA","SA","QLD"],
    "SA":  ["WA","NT","QLD","NSW","VIC"],
    "QLD": ["NT","SA","NSW"],
    "NSW": ["SA","QLD","VIC"],
    "VIC": ["SA","NSW",],
    "TAS": [],
}

# Maps color-number strings to real display colors for visualization
COLOR_PALETTE = {
    "1": "#E63946",   # red
    "2": "#457B9D",   # blue
    "3": "#2A9D8F",   # teal
    "4": "#E9C46A",   # yellow
    "5": "#F4A261",   # orange
}

# Maps AU abbreviations to the name sused in the shape files for visualization
AU_ABBR_TO_NAME = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA":  "South Australia",
    "WA":  "Western Australia",
    "TAS": "Tasmania",
    "NT":  "Northern Territory",
}



# VISUALIZATION HELPERS
# Used to reposition AK and HI for visualization
def translate_geometries(df, x, y, scale, rotate):
    '''
    Translate, scale, and rotate geometries.

    :param df: GeoDataFrame to transform
    :param x: x-axis translation offset
    :param y: y-axis translation offset
    :param scale: scale factor
    :param rotate: rotation angle (degrees)
    '''
    df.loc[:, "geometry"] = df.geometry.translate(yoff=y, xoff=x)
    center = df.dissolve().centroid.iloc[0]
    df.loc[:, "geometry"] = df.geometry.scale(xfact=scale, yfact=scale, origin=center)
    df.loc[:, "geometry"] = df.geometry.rotate(rotate, origin=center)
    return df


def adjust_us_map(df):
    '''
    Reposition Alaska and Hawaii into position below the
    contiguous 48 states so the full map fits in one frame.

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
    Render a colored map

    :param region: "us" or "au"
    :param coloring: dict mapping region abbreviationto hex color string,
                     e.g. {"CA": "#E63946", "TX": "#457B9D"}
    '''
    edge_color       = "#30011E"
    background_color = "#fafafa"

    sns.set_style({
        "font.family": "serif",
        "figure.facecolor": background_color,
        "axes.facecolor":   background_color,
    })

    # Sets appropriate variables and gets data for given region
    if region == "us":
        gdf = gpd.read_file("./map_data/us/")
        gdf = gdf.to_crs("ESRI:102003")
        key_col = "STUSPS"         
        
        # Remove US territories from map
        gdf = gdf[~gdf.STATEFP.isin(["72", "69", "60", "66", "78"])]
        gdf = adjust_us_map(gdf) # Move AK+HI


    elif region == "au":
        gdf = gpd.read_file("./map_data/au/")
        gdf = gdf.to_crs("EPSG:3577")
        key_col = "STE_NAME21"

        # Remove AU territories from map
        gdf = gdf[~gdf[key_col].isin(
            ["Other Territories", "Outside Australia", "Australian Capital Territory"]
        )]

        # Convert abbreviation keys to the full state names used in the shapefile
        if coloring:
            coloring = {AU_ABBR_TO_NAME.get(k, k): v for k, v in coloring.items()}

    else:
        raise ValueError("region must be 'us' or 'au'")

    # Apply colors to regions; fall back to grey for any uncolored region
    if coloring:
        gdf["plot_color"] = gdf[key_col].map(coloring).fillna("#dddddd")
    else:
        gdf["plot_color"] = "#dddddd"

    # Plotting each region with the defined color
    ax = gdf.plot(
        edgecolor=edge_color,
        color=gdf["plot_color"],
        linewidth=1,
    )

    plt.axis("off")
    plt.title(f"{region.upper()} Map Coloring", fontsize=14)
    
    filename = f"{region}_map.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def assignment_to_hex_coloring(assignment):
    '''
    Convert an assignment dict (region: number) into one with
    colors for plot_map visualization (region: hex color string).

    :assignment: dict[str, str]
        ex/ {"CA": "1", "NV": "2"}
    '''
    return {region: COLOR_PALETTE.get(color, "#dddddd")
            for region, color in assignment.items()}



# CSP Class
class MapColoringProblem:
    '''
    Defines a map-coloring CSP.

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

    # constraint check 
    def is_consistent(self, var, color, assignment):
        '''
        Checks if a color assignment to a var conflicts with any already
        assigned variables/regions.

        :param var: variable/region being assigned
        :param color: candidate color
        :param assignment: dict of already-assigned variables/regions
        '''
        for neighbor in self.neighbors[var]:
            if assignment.get(neighbor) == color:
                return False
        return True
    

    def forward_check(self, var, color, domains, assignment):
        '''
        Remove a color from every unassigned neighbor's domain.
        If any neighbor ends up with no valid colors left, return False.

        :param var: variable/region just assigned
        :param color: color just assigned to var
        :param domains: mutable copy of current domains
        :param assignment: list of all assignments
        '''
        for neighbor in self.neighbors[var]:
            if neighbor not in assignment and color in domains[neighbor]:
                domains[neighbor].remove(color)

                if not domains[neighbor]:
                    return False   
        return True
    
    
    def propagate_singletons(self, domains):
        '''
        Keep enforcing constraints for variables/regions that only have one
        possible color left.

        If a variable is forced to a single color, remove that color
        from its neighbors. Repeat until nothing changes.

        Returns False if any domain becomes empty, otherwise True.

        :param domains: mutable copy of current domains
        '''
        changed = True
        while changed:
            changed = False
            for var in self.variables:
                # For each variable/region, check if theres only one color in the domain   
                if len(domains[var]) == 1:
                    forced_color = domains[var][0]
                    # Remove that one color from the domains of other neighboard
                    for neighbor in self.neighbors[var]:
                        if forced_color in domains[neighbor]:
                            domains[neighbor].remove(forced_color)
                            changed = True

                            # If empty domain
                            if not domains[neighbor]:
                                return False 
        return True

    # HEURISTICS
    def mrv(self, unassigned, domains):
        '''
        Minimum Remaining Values
        Picks the variable with the fewest remaining color options.

        :param unassigned: list of not-yet-assigned variables
        :param domains: current domains dict
        '''
        return min(unassigned, key=lambda v: len(domains[v]))


    def degree(self, unassigned):
        '''
        Degree heuristic
        Picks the variable connected to the most other unassigned variables.

        :param unassigned: list of not-yet-assigned variables
        '''
        unassigned_set = set(unassigned)
        return max(unassigned,
                   key=lambda v: sum(1 for n in self.neighbors[v]
                                     if n in unassigned_set))


    def lcv(self, var, domains, assignment):
        '''
        Least Constraining Value
        Order var (colors) such that those in the least domains 
        (least restrictive) come first.        

        :param var: region/variable being assigned
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
        Choose the next region/variable to assign.

        With heuristics: use mrv, break ties with degree.
        Without: return the next region/variable in the preset order.

        :param unassigned: ordered list of unassigned regions/variables
        :param domains: current domains/possible colors dict
        :param use_heuristics: bool
        '''
        if not use_heuristics:
            return unassigned[0]

        min_domain = min(len(domains[v]) for v in unassigned)
        mrv_candidates = [v for v in unassigned if len(domains[v]) == min_domain]

        if len(mrv_candidates) == 1:
            return mrv_candidates[0]

        return self.degree(mrv_candidates)  # tie-break


    def order_domain_values(self, var, domains, assignment, use_heuristics):
        '''
        Decide what order to try var (colors) for a variable/region.

        With heuristics: LCV ordering.
        Without: domain as-is.

        :param var: region/variable being assigned
        :param domains: current domains dict
        :param assignment: current partial assignment
        :param use_heuristics: bool
        '''
        if use_heuristics:
            return self.lcv(var, domains, assignment)
        
        return list(domains[var])



def backtrack(problem, assignment, unassigned, domains,
              use_fc, use_propagation, use_heuristics):
    '''
    Recursive backtracking search.

    Tries assigning values(colors) to variables(regions) one at a time, and
    backtracks when a conflict or dead end is reached.

    :param problem: MapColoringProblem instance
    :param assignment: dict mapping region:color
    :param unassigned: list of variables/regions not yet assigned
    :param domains: dict of current legal colors per variable
    :param use_fc: enable forward checking
    :param use_propagation: enable singleton-domain propagation
    :param use_heuristics: enable MRV / Degree / LCV heuristics
    '''
    if not unassigned: # If unassaigned list is empty, the assignment is complete
        return assignment

    # Select the next region to be assigned and take it out of the remaining list
    var = problem.select_unassigned_variable(unassigned, domains, use_heuristics)
    remaining = [v for v in unassigned if v != var]

    # Order colors in the domain depending on heuristics and assign them to 
    # variables selected.
    for color in problem.order_domain_values(var, domains, assignment, use_heuristics):
        if problem.is_consistent(var, color, assignment):
            assignment[var] = color
            domains[var] = [color]

            # Save domains before modification so we can restore on failure
            saved_domains = {v: list(d) for v, d in domains.items()}
            ok = True

            if use_fc: # Checking if assignment passes the forward check
                ok = problem.forward_check(var, color, domains, assignment)
        

            if ok and use_propagation: # Checking if singletons cause failure
                ok = problem.propagate_singletons(domains)

             # If fc+singleton pass/aren't used, regressively assign color to the next region
            if ok:
                result = backtrack(problem, assignment, remaining, domains,
                                   use_fc, use_propagation, use_heuristics)
                
                # Passing the completed assignment to the caller
                if result is not None:
                    return result

            # The color assignment failed to meet constraints, backtrack
            # Restore domains and undo assignment
            for v in domains:
                domains[v] = saved_domains[v]

            del assignment[var]

            problem.backtracks += 1

    return None # signal backtrack to caller


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
    problem = MapColoringProblem(neighbors, num_colors)
    problem.variables = list(variable_order)
    domains = {v: list(problem.colors) for v in problem.variables}

    start = time.perf_counter()
    assignment = backtrack(problem, {}, list(problem.variables), domains,
                           use_fc, use_propagation, use_heuristics)
    elapsed = (time.perf_counter() - start) * 1000 # ms

    return assignment, problem.backtracks, elapsed


def find_chromatic_number(neighbors, max_colors=10):
    '''
    Find the minimum number of colors needed to legally color the graph
    by increasing color counts until a solution is found.

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



# Run experiement and print results
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
        print(f"{'Trial':<7} | {'Algorithm':<20} | {'Backtracks':>10} | {'Time (ms)':>10}")
        print(f"{'-'*7}-+-{'-'*20}-+-{'-'*10}-+-{'-'*20}")

        for trial in range(1, num_trials + 1):
            random.shuffle(variables)
            order = list(variables)

            for label, (fc, prop, heur) in zip(ALGORITHM_LABELS, configs):
                assignment, bt, t = solve(neighbors, num_colors, order, fc, prop, heur)
                status = "✓" if assignment else "✗"
                print(f"{trial:<7} | {label:<20} | {bt:>10} | {t:>10.5f} ms  {status}")

            print(f"{'-'*7}-+-{'-'*20}-+-{'-'*10}-+-{'-'*20}")


def print_coloring(assignment, map_name):
    '''
    Print a coloring assignment to the terminal.

    :param assignment: dict mapping region → color-number string
    :param map_name: label shown in the header
    '''
    # Group states by color
    color_groups = defaultdict(list)
    for region, color in assignment.items():
        color_groups[color].append(region)

    # Sort colors numerically ("1", "2", ...)
    sorted_colors = sorted(color_groups.keys(), key=int)

    print(f"\n{map_name} coloring:")
    print(f"{'Color':<10} | States")
    print(f"{'-'*10}-+-{'-'*60}")

    for color in sorted_colors:
        states = sorted(color_groups[color])
        state_str = ", ".join(states)
        print(f"{color:<10} | {state_str}")




if __name__ == "__main__":
    random.seed(42)

    with open("experiment_results.txt", "w", encoding="utf-8") as f:
        sys.stdout = f   # Print everything to a txt file
        
        # Get chromatic numbers from US and AU
        print("Computing chromatic numbers …")
        chi_usa = find_chromatic_number(USA_NEIGHBORS)
        print(f"  USA chromatic number  χ(G) = {chi_usa}")

        chi_au = find_chromatic_number(AU_NEIGHBORS)
        print(f"  Australia chromatic number  χ(G) = {chi_au}")

        # Run experiements on each map
        # chi + 1 keeps every trial tractable while still exercising all variants
        run_experiments(USA_NEIGHBORS, "USA MAP", chi_usa + 1, num_trials=10)
        run_experiments(AU_NEIGHBORS, "AUSTRALIA MAP", chi_au + 1, num_trials=10)

        # Solve once for visualization
        usa_sol, _, _ = solve(USA_NEIGHBORS, chi_usa, list(USA_NEIGHBORS.keys()),
                            use_fc=True, use_propagation=True, use_heuristics=True)
        au_sol,  _, _ = solve(AU_NEIGHBORS, chi_au, list(AU_NEIGHBORS.keys()),
                            use_fc=True, use_propagation=True, use_heuristics=True)

        if usa_sol:
            print_coloring(usa_sol, "USA")
            plot_map("us",  assignment_to_hex_coloring(usa_sol))

        if au_sol:
            print_coloring(au_sol, "Australia")
            plot_map("au",  assignment_to_hex_coloring(au_sol))

    sys.stdout = sys.__stdout__  # restore normal printing