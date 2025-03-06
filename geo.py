#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
# muzete pridat vlastni knihovny
from shapely.geometry import box
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Konvertuje pandas DataFrame do geopandas GeoDataFrame se správným kódováním.
    
    :param df: DataFrame s daty nehod.
    :return: GeoDataFrame s daty nehod a geometrií bodů.
    """

    # odstraneni radku
    df = df.dropna(subset=['d', 'e'])

    # sloupec pro datum
    df.loc[:, "date"] = pd.to_datetime(df["p2a"], cache=True)

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['d'], df['e']))

    czech_republic_bounds = box(-951499.37, -1353292.51, -159365.31, -911053.67)

    gdf = geopandas.clip(gdf, czech_republic_bounds)

    # nastaveni crs
    gdf.crs = "EPSG:5514"

    return gdf

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Vykresluje graf s nehodami pro kraj "JHM" způsobenými zvěří.
    
    :param gdf: GeoDataFrame s daty nehod.
    :param fig_location: Cesta k souboru pro uložení grafu.
    :param show_figure: Zda se má graf zobrazit.
    """
    
    # filtrovani kraje a nehod se zveri
    gdf = gdf[(gdf['region'] == 'JHM') & (gdf['p10'] == 4)]
    
    # rozdeleni na roky
    gdf_2021 = gdf[gdf['date'].dt.year == 2021]
    gdf_2022 = gdf[gdf['date'].dt.year == 2022]
    
    # prevod do webmercator
    gdf_2021 = gdf_2021.to_crs(epsg=3857)
    gdf_2022 = gdf_2022.to_crs(epsg=3857)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))  # Změna na 1 řádek, 2 sloupce
    
    gdf_2021.plot(ax=axs[0], color='red', markersize=5)
    axs[0].set_title('JHM kraj (2021)')
    axs[0].set_axis_off()
    ctx.add_basemap(axs[0], source=ctx.providers.OpenStreetMap.Mapnik)
    
    gdf_2022.plot(ax=axs[1], color='red', markersize=5)
    axs[1].set_title('JHM kraj (2022)')
    axs[1].set_axis_off()
    ctx.add_basemap(axs[1], source=ctx.providers.OpenStreetMap.Mapnik)
    
    if fig_location:
        plt.savefig(fig_location)
    
    if show_figure:
        plt.show()

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Vykresluje graf s lokalitou všech nehod v kraji shlukovaných do clusterů.
    
    :param gdf: GeoDataFrame s daty nehod.
    :param fig_location: Cesta k souboru pro uložení grafu.
    :param show_figure: Zda se má graf zobrazit.
    """
    
    # filtrovani podle kraje a sloupce p11
    gdf = gdf[(gdf['region'] == 'JHM') & (gdf['p11'] >= 4)].to_crs(epsg=3857)
    
    # vytvoreni clusteru
    agg = sklearn.cluster.AgglomerativeClustering(n_clusters=12).fit(gdf[['d', 'e']])
    gdf['cluster'] = agg.labels_

    gdf = gdf.dissolve(by='cluster', aggfunc={'p1': 'count', 'cluster': 'first'})

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # podbarveni clusteru
    for cluster in gdf['cluster'].unique():
        gdf[gdf['cluster'] == cluster].convex_hull.plot(ax=ax, color='grey', alpha=0.5)
    
    gdf.plot(ax=ax, markersize=1, column="p1", legend=True)
    ax.set_axis_off()
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    plt.title("Nehody v JHM kraji s významnou měrou alkoholu")

    plt.tight_layout()
    
    if fig_location:
        plt.savefig(fig_location)
    
    if show_figure:
        plt.show()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
