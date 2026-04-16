'''
function to print out the colored map given the region and a dict mapping the state to a color
'''

import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

AU_ABBR_TO_NAME = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "WA": "Western Australia",
    "TAS": "Tasmania",
    "NT": "Northern Territory"
}

def translate_geometries(df, x, y, scale, rotate):
    df.loc[:, "geometry"] = df.geometry.translate(yoff=y, xoff=x)
    center = df.dissolve().centroid.iloc[0]
    df.loc[:, "geometry"] = df.geometry.scale(xfact=scale, yfact=scale, origin=center)
    df.loc[:, "geometry"] = df.geometry.rotate(rotate, origin=center)
    return df

def adjust_us_map(df):
    df_main = df[~df.STATEFP.isin(["02", "15"])]
    df_alaska = df[df.STATEFP == "02"]
    df_hawaii = df[df.STATEFP == "15"]

    df_alaska = translate_geometries(df_alaska, 1300000, -4900000, 0.5, 32)
    df_hawaii = translate_geometries(df_hawaii, 5400000, -1500000, 1, 24)

    return pd.concat([df_main, df_alaska, df_hawaii])


def plot_map(region="us", coloring=None):
    """
    :param region: String; "us" or "au"
    :param coloring: dict mapping region name/code -> color
              e.g. {"CA": "red", "TX": "blue"} or {"NSW": "green"}
    """

    edge_color = "#30011E"
    background_color = "#fafafa"

    sns.set_style({
        "font.family": "serif",
        "figure.facecolor": background_color,
        "axes.facecolor": background_color,
    })

    # Load data
    if region == "us":
        gdf = gpd.read_file("./map_data/us/")
        gdf = gdf[~gdf.STATEFP.isin(["72", "69", "60", "66", "78"])]
        gdf = gdf.to_crs("ESRI:102003")
        gdf = adjust_us_map(gdf)

        key_col = "STUSPS"  # e.g. CA, TX, NY

    elif region == "au":
        gdf = gpd.read_file("./map_data/au/")
        gdf = gdf.to_crs("EPSG:3577")

        key_col = "STE_NAME21"

        # remove unwanted regions
        gdf = gdf[~gdf[key_col].isin(["Other Territories", "Outside Australia", "Australian Capital Territory"])]

        # convert abbreviation coloring → full names
        if coloring:
            coloring = {
                AU_ABBR_TO_NAME.get(k, k): v
                for k, v in coloring.items()
            }

        print(set(gdf[key_col]))

    else:
        raise ValueError("region must be 'us' or 'au'")

    # apply colors
    if coloring:
        gdf["plot_color"] = gdf[key_col].map(coloring).fillna("#dddddd")
    else:
        gdf["plot_color"] = "#dddddd"

    # plot
    ax = gdf.plot(
        edgecolor=edge_color,
        color=gdf["plot_color"],
        linewidth=1
    )

    plt.axis("off")
    plt.title(f"{region.upper()} Map Coloring", fontsize=14)
    plt.show()

# if __name__ == "__main__":
    # us_coloring = {
    #     "CA": "red",
    #     "NV": "green",
    #     "AZ": "blue",
    #     "UT": "red",
    #     "TX": "green"
    # }

    # plot_map("us", us_coloring)

    # au_coloring = {
    #     "WA": "red",
    #     "NT": "green",
    #     "SA": "blue",
    #     "QLD": "red",
    #     "NSW": "green",
    #     "VIC": "red",
    #     "TAS": "blue"
    # }

    # plot_map("au", au_coloring)