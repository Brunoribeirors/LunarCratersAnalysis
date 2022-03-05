#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:06:27 2022

@author: brunomr
"""

# Importando BD Pandas
from mpl_toolkits.basemap import Basemap
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from geopandas import GeoDataFrame
import geopandas as gpd
from dask.distributed import Client
import dask.dataframe as dd
from matplotlib import pyplot as plt
from math import log10
from matplotlib import pyplot
from scipy.optimize import curve_fit
from pandas import read_csv
from numpy import arange
from numpy import sqrt
from numpy import sin
from irfpy.moon import moon_map
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import fiona
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Point
from datetime import datetime
import geopandas
import numpy as np
import pandas as pd
craters = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv')
print(craters)
print()
print(craters.loc[12])
print()

# Itens faltantes por atributo
percent_missing = craters.isnull().sum() * 100 / len(craters)
percent_missing = percent_missing.sort_values(ascending=False)
print(percent_missing)
print()

print('Contagem de crateras por quadrantes de Longitude')
print(craters['LON_CIRC_IMG'].value_counts(bins=4, sort=False))
print()

print('Contagem de crateras por diâmetro')
print(craters['DIAM_CIRC_IMG'].value_counts(bins=20, sort=False))
print()

###################################################################
# Moon geologic regions
conda install geopandas
craters_diam = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_regions.csv')
print(craters_diam)

craters_diam = craters_diam.sort_values('DIAM_CIRC_IMG', ascending=False)
craters_diam
craters_diam.reset_index(drop=True, inplace=True)


def crater_region(lat, lon):
    lat = lat * 30323.34
    lon = (lon - 180) * 30323.34
    result = find_region(lat, lon)
    return result


def find_region(lat, lon):
    shape_df = geopandas.read_file(
        '/home/brunomr/Documents/TCC/Unified_Geologic_Map_of_the_Moon_GIS_v2/Lunar_GIS/Shapefiles/GeoUnits.shp')
    for index, row in shape_df.iterrows():
        poly = np.array(row['geometry'])
        point = Point(lon, lat)
        if point.within(row['geometry']) == True:
            region = row['FIRST_Unit']
    return region


craters_diam['REGION'] = 0

% % time
for idx, crater in craters_diam.iterrows():
    if idx < 725000:
        pass
    else:
        craters_diam.loc[idx, 'REGION'] = crater_region(
            crater['LAT_CIRC_IMG'], crater['LON_CIRC_IMG'])
    if idx % 5000 == 0:
        craters_diam.to_csv(
            r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_regions.csv', index=False)

# Clusterização Cratera dentro de cratera
def distance(lat1, lat2, lon1, lon2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    #r = 6371
    r = 1737
    return(c * r)
#dist = distance(0, 0, 0, 1)


craters['CLUSTER'] = 0
craters_diam = craters.sort_values('DIAM_CIRC_IMG', ascending=False)
craters_diam
craters_diam.reset_index(drop=True, inplace=True)

start = datetime.now()
cluster = 1
for i in range(len(craters_diam)):
    start2 = datetime.now()
    if craters_diam.loc[i, 'CLUSTER'] == 0:
        craters_diam.loc[i, 'CLUSTER'] = cluster
        lat1 = craters_diam.loc[i, 'LAT_CIRC_IMG']
        lon1 = craters_diam.loc[i, 'LON_CIRC_IMG']
        diam = craters_diam.loc[i, 'DIAM_CIRC_IMG']
        for j in range(i+1, len(craters_diam)):
            if craters_diam.loc[j, 'CLUSTER'] == 0:
                lat2 = craters_diam.loc[j, 'LAT_CIRC_IMG']
                lon2 = craters_diam.loc[j, 'LON_CIRC_IMG']
                dist = distance(lat1, lat2, lon1, lon2)
                if dist <= diam/2:
                    craters_diam.loc[j, 'CLUSTER'] = cluster
        cluster += 1
    print(datetime.now() - start2)
    if cluster % 500 == 0:
        craters_diam.to_csv(
            r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_clusters_part.csv', index=False)
print(datetime.now() - start)

craters_diam.to_csv(
    r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_clusters.csv', index=False)

craters_diam = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_clusters.csv')
craters_diam

#############################################################################
# Clusterização numpy
craters_np = craters_diam[['LAT_CIRC_IMG',
                           'LON_CIRC_IMG', 'DIAM_CIRC_IMG', 'CLUSTER']].to_numpy()
craters_np[:, 3]
len(craters_np)


def distance(lat1, lat2, lon1, lon2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    #r = 6371
    r = 1737
    return(c * r)


start = datetime.now()
cluster = 1
for i in range(len(craters_np)):
    start2 = datetime.now()
    if craters_np[i, 3] == 0:
        craters_np[i, 3] = cluster
        lat1 = craters_np[i, 0]
        lon1 = craters_np[i, 1]
        diam = craters_np[i, 2]
        for j in range(i+1, len(craters_np)):
            if craters_np[j, 3] == 0:
                lat2 = craters_np[j, 0]
                lon2 = craters_np[j, 1]
                dist = distance(lat1, lat2, lon1, lon2)
                if dist <= diam/2:
                    craters_np[j, 3] = cluster
        cluster += 1
    print(datetime.now() - start2)
    # if cluster%500 == 0:
    #craters_diam.to_csv(r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_clusters_part.csv', index=False)
print(datetime.now() - start)

craters_diam['CLUSTER_NP'] = craters_np[:, 3]

########################################################################
#Usando Dask Dataframe
import dask.dataframe as dd
craters_dd = dd.read_csv('/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv')
print(craters_dd.head())

craters_diam_dd = craters_dd.sort_values('DIAM_CIRC_IMG', ascending=False)
craters_diam_dd.head()
craters_diam_dd.reset_index(drop=True)#, inplace=True)
clusters = [0] * len(craters_dd)

from math import radians, cos, sin, asin, sqrt
def distance(lat1, lat2, lon1, lon2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    #r = 6371
    r = 1737
    return(c * r)
#dist = distance(0, 0, 0, 1)

from datetime import datetime
start = datetime.now()
cluster = 1
cont = 0
for crater in craters_diam_dd.iterrows():  
    start2 = datetime.now()
    if clusters[cont] == 0:
        clusters[cont] = cluster
        lat1 = crater[1][1]
        lon1 = crater[1][2]
        diam = crater[1][5]
        for j in range(cont+1, len(craters_diam_dd)):
            if clusters[j] == 0:
                crater2 = next(craters_diam_dd.iterrows())[j]
                lat2 = craters_diam_dd[j]
                lon2 = craters_diam_dd.loc[j, 'LON_CIRC_IMG']
                #dist = sqrt((lat - lat2) ** 2 + (lon - lon2) ** 2)
                dist = distance(lat1, lat2, lon1, lon2)
                #if dist*30.32335 <= diam/2:
                if dist <= diam/2:
                    clusters[j] = cluster
        cluster += 1
    print(datetime.now() - start2)
print(datetime.now() - start)

craters_diam_dd.dtypes
len(craters_diam_dd)
craters_diam_dd.head()
craters_diam_dd.shape()
craters_diam_dd.loc[12, 'CLUSTER']
row = next(craters_diam_dd.iterrows())[1]  
row  

##############################################################################
import dask.dataframe as dd
craters_dd = dd.read_csv('/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv')
print(craters_dd.head())
craters_dd
craters_dd.partitions[1].compute()
craters_dd.map_partitions(len).compute()

craters_dd = dd.from_pandas(craters, npartitions=4)
crater = next(craters_dd.iterrows())
crater

#conda install -c conda-forge fiona
import fiona
import numpy as np
import matplotlib.path as mplPath

craters_dd['FU1'] = 0
craters_dd['FU2'] = 0
craters_dd['FUnit'] = 0
FU1 = list()
FU2 = list()
FUnit = list()

from datetime import datetime
start = datetime.now()
for crater in craters_dd.iterrows():  
    start2 = datetime.now()
    lat = crater[1][1]
    lat = lat * 30323.34
    lon = crater[1][2] - 180
    lon = lon * 30323.34
    shape = fiona.open('/home/brunomr/Documents/TCC/Unified_Geologic_Map_of_the_Moon_GIS_v2/Lunar_GIS/Shapefiles/GeoUnits.shp')
    for j in range(len(shape)):
        shp = shape.next()
        geo = shp['geometry']['coordinates']
        for k in range(len(geo)):
            array = np.array(geo[k])
            size = len(geo[k])
            array = array.reshape(size,2)
            poly_path = mplPath.Path(np.array(array))
            point = (lon, lat)
            if poly_path.contains_point(point) == True:
                FU1.append(shp['properties']['FIRST_Un_1'])
                FU2.append(shp['properties']['FIRST_Un_2'])
                FUnit.append(shp['properties']['FIRST_Unit'])
    print(datetime.now() - start2)
    #if i%1000 == 0:
        #craters_diam.to_csv(r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_clusters_part2.csv', index=False)
print(datetime.now() - start)

##############################################################################
# Arquivo completo Cluster/Region
craters_diam_c = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_clusters_part.csv')
craters_diam_c

craters_diam_r = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_regions.csv')
craters_diam_r

craters_diam_c['REGION'] = craters_diam_r['REGION']
craters_diam = craters_diam_c

craters_diam.to_csv(
    r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_cr.csv', index=False)
craters_diam = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_cr.csv')
craters = pd.read_csv(
    '/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_complete.csv')


# Anállise Exploratória inicial
# Plotando Gráficos
import pandas as pd
craters = pd.read_csv(
    r'/home/brunomr/Documents/TCC/lunar_crater_database_robbins_2018_bundle/data/lunar_craters_complete.csv')

result = craters.groupby(by='CLUSTER', as_index=False).agg(
    {'CRATER_ID': pd.Series.nunique})
print(result)

craters_quarter = craters['LON_CIRC_IMG'].value_counts(bins=4, sort=False)
craters_quarter_barplot = sns.barplot(
    x=craters_quarter.keys(), y=craters_quarter)
craters_quarter_barplot.set_xticklabels(
    craters_quarter_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_quarter_barplot.set_title("Barplot: Crateras por quarto de Longitude")
craters_quarter_barplot

craters_quarter = craters['LON_CIRC_IMG'].value_counts(bins=36, sort=False)
craters_quarter_barplot = sns.barplot(
    x=craters_quarter.keys(), y=craters_quarter)
craters_quarter_barplot.set_xticklabels(
    craters_quarter_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_quarter_barplot.set_title("Barplot: Crateras por 10° de Longitude")
craters_quarter_barplot

# Limpando crateras maiores de 200km
craters_H_100 = craters[craters['DIAM_CIRC_IMG'] >= 200]
print(craters_H_100['DIAM_CIRC_IMG'].idxmax())
print()
craters_H_100 = craters_H_100.drop({12})
print(craters_H_100['DIAM_CIRC_IMG'].value_counts(bins=20, sort=False))
print()

craters_diam = craters_H_100['DIAM_CIRC_IMG'].value_counts(bins=20, sort=False)
craters_diam_barplot = sns.barplot(x=craters_diam.keys(), y=craters_diam)
craters_diam_barplot.set_xticklabels(
    craters_diam_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_diam_barplot.set_title("Barplot: Diâmetro das Crateras maiores 200km")
craters_diam_barplot

# PLot excentricidade
craters_eccen = craters['DIAM_ELLI_ECCEN_IMG'].value_counts(
    bins=100, sort=False)
craters_eccen_barplot = sns.barplot(x=craters_eccen.keys(), y=craters_eccen)
craters_eccen_barplot.set_xticklabels(
    craters_eccen_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_eccen_barplot.set_title("Barplot: Excentricidade")
craters_eccen_barplot

# Crateras redondas
craters_red = craters[(craters['DIAM_ELLI_ECCEN_IMG'] < 0.30)]
print(craters_red)

# Crateras elípticas
craters_elli = craters[(craters['DIAM_ELLI_ECCEN_IMG'] > 0.70)]
print(craters_elli)

craters_lon = craters_red['LON_CIRC_IMG'].value_counts(bins=36, sort=False)
craters_lon_barplot = sns.barplot(x=craters_lon.keys(), y=craters_lon)
craters_lon_barplot.set_xticklabels(
    craters_lon_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_lon_barplot.set_title("Barplot: Crateras redondas por Longitude")
craters_lon_barplot
print("Crateras redondas por Longitude")
print(craters_red['LON_CIRC_IMG'].value_counts(bins=36, sort=False))
print()

craters_lon = craters_elli['LON_CIRC_IMG'].value_counts(bins=36, sort=False)
craters_lon_barplot = sns.barplot(x=craters_lon.keys(), y=craters_lon)
craters_lon_barplot.set_xticklabels(
    craters_lon_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_lon_barplot.set_title("Barplot: Crateras elípticas por Longitude")
craters_lon_barplot
print("Crateras elípticas por Longitude")
print(craters_elli['LON_CIRC_IMG'].value_counts(bins=36, sort=False))
print()

craters_lat = craters_red['LAT_CIRC_IMG'].value_counts(bins=18, sort=False)
craters_lat_barplot = sns.barplot(x=craters_lat.keys(), y=craters_lat)
craters_lat_barplot.set_xticklabels(
    craters_lat_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_lat_barplot.set_title("Barplot: Crateras redondas por Latitude")
craters_lat_barplot
print("Crateras redondas por Latitude")
print(craters_red['LAT_CIRC_IMG'].value_counts(bins=18, sort=False))
print()

craters_lat = craters_elli['LAT_CIRC_IMG'].value_counts(bins=18, sort=False)
craters_lat_barplot = sns.barplot(x=craters_lat.keys(), y=craters_lat)
craters_lat_barplot.set_xticklabels(
    craters_lat_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')
craters_lat_barplot.set_title("Barplot: Crateras elípticas por Latitude")
craters_lat_barplot
print("Crateras elípticas por Latitude")
print(craters_elli['LAT_CIRC_IMG'].value_counts(bins=18, sort=False))
print()

# Modelagem de Distribuição Cumulativa de Crateras por Diâmetro
# Areas das regioes geologicas
shape = fiona.open(
    '/home/brunomr/Documents/TCC/Unified_Geologic_Map_of_the_Moon_GIS_v2/Lunar_GIS/Shapefiles/GeoUnits.shp')
regions = set()
for i in range(0, len(shape)):
    shp = shape.next()
    region = shp['properties']['FIRST_Unit']
    regions.add(region)

areas = {}
for region in regions:
    area = 0
    shape = fiona.open(
        '/home/brunomr/Documents/TCC/Unified_Geologic_Map_of_the_Moon_GIS_v2/Lunar_GIS/Shapefiles/GeoUnits.shp')
    for i in range(0, len(shape)):
        shp = shape.next()
        if region == shp['properties']['FIRST_Unit']:
            area += shp['properties']['AREA_GEO']
    areas[region] = area
print(areas)

# Medindo Populações de Crateras
# Regiões de Pouso de missões/sondas
apollos_landing_sites = [[00.67408, 23.47297], [-3.01239,  -23.42157], [-3.64530, -17.47136],
                         [26.1322, 3.6339], [-8.97301,
                                             15.50019], [20.1908, 30.7717],
                         [-0.5137, 56.3638], [3.7863, 56.6242], [12.7145, 62.2097], [43.06, -51.92]]


sites = []
for i in range(0, len(apollos_landing_sites)):
    lat = apollos_landing_sites[i][0]
    lat = lat * 30323.34
    lon = apollos_landing_sites[i][1]
    lon = lon * 30323.34
    shape = fiona.open(
        '/home/brunomr/Documents/TCC/Unified_Geologic_Map_of_the_Moon_GIS_v2/Lunar_GIS/Shapefiles/GeoUnits.shp')
    for j in range(len(shape)):
        shp = shape.next()
        geo = shp['geometry']['coordinates']
        for k in range(len(geo)):
            array = np.array(geo[k])
            size = len(geo[k])
            array = array.reshape(size, 2)
            poly_path = mplPath.Path(np.array(array))
            point = (lon, lat)
            if poly_path.contains_point(point) == True:
                apollos_landing_sites[i].append(
                    shp['properties']['FIRST_Unit'])
apollos_landing_sites[1].pop()
apollos_landing_sites[1].pop()
apollos_landing_sites[2].pop()
apollos_landing_sites[2].pop()


craters_hist = craters[(craters['REGION'] == 'Nt')]
D = craters_hist['DIAM_CIRC_IMG'].value_counts(bins=1000, sort=False)
N = D.index
D = np.flipud(np.flipud(D).cumsum())
I = []
for k in range(len(N)):
    I.append(N[k].right)
D = D/areas['Nt']
D = np.log10(D)
I = np.log10(I)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(0, 5))
plt.scatter(I, D)
plt.plot(I, D)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('D (km)', fontsize=20)
plt.ylabel('N (/km²)', fontsize=20)
plt.title(
    'Distribuição cumulativa de Frequência Crateras - Nectarian Terra', fontsize=20)
plt.show()

# Equação Polinominal
D = np.log10(D)
I = np.log10(I)


def get_equation(x, y, d):
    degree = d
    coefs, res, _, _, _ = np.polyfit(x, y, degree, full=True)
    ffit = np.poly1d(coefs)
    print(ffit)
    return ffit


eq = get_equation(I, D, 10)

x = np.linspace(0.2, 2.8, 100)
Im2 = -0.4545*x**8 + 4.648*x**7 - 18.77*x**6 + 37.64*x**5 - \
    37.85*x**4 + 14.9*x**3 + 2.742*x**2 - 5.428*x - 6.396
plt.plot(x, Im2)
x = np.linspace(0.1, 2.4, 100)
Nb = -1.263*x**9 + 11.36*x**8 - 39.9*x**7 + 66.54*x**6 - 44.68 * \
    x**5 - 14.38*x**4 + 41.69*x**3 - 24.54*x**2 + 3.536*x - 7.877
plt.plot(x, Nb)
x = np.linspace(0.2, 2.7, 100)
Ip = -0.03636*x**6 + 0.9389*x**5 - 5.423*x**4 + \
    12.73*x**3 - 13.47*x**2 + 3.974*x - 7.882
plt.plot(x, Ip)
x = np.linspace(0.1, 2.4, 100)
Iif = 0.7802*x**6 - 5.455*x**5 + 14.26*x**4 - \
    17.27*x**3 + 9.665*x**2 - 4.031*x - 6.983
plt.plot(x, Iif)
x = np.linspace(0.2, 2.8, 100)
Em = -0.404*x**7 + 3.903*x**6 - 14.78*x**5 + 27.78 * \
    x**4 - 27.16*x**3 + 13.28*x**2 - 4.895*x - 6.747
plt.plot(x, Em)

x = np.linspace(0.2, 1.7, 100)
fig = plt.figure(figsize=(10, 5))
# Create the plot
plt.plot(x, Em)
plt.plot(x, Im2)
plt.plot(x, Iif)
plt.plot(x, Ip)
plt.plot(x, Nb)

# Show the plot
plt.show()

# Cálculo de diferença entre 2 equações
landing_regions = ['Em', 'Im2', 'Iif', 'Ip', 'Nb']
for i in range(0, len(landing_regions)):
    if landing_regions[i] == 'Em':
        q = Em
    elif landing_regions[i] == 'Im2':
        q = Im2
    elif landing_regions[i] == 'Iif':
        q = Iif
    elif landing_regions[i] == 'Ip':
        q = Ip
    elif landing_regions[i] == 'Nb':
        q = Nb
    for j in range(0, len(landing_regions)):
        if landing_regions[j] == 'Em':
            w = Em
        elif landing_regions[j] == 'Im2':
            w = Im2
        elif landing_regions[j] == 'Iif':
            w = Iif
        elif landing_regions[j] == 'Ip':
            w = Ip
        elif landing_regions[j] == 'Nb':
            w = Nb
        y = q-w
        r = y
        for i in range(0, len(y)):
            r[i] = y[i]/q[i]
        print(r.mean())
        print(np.median(r))

# Calibração
# Im2
ages = [2.86, 3.4, 3.5, 3.75, 3.9, 4.33, 4.42]
em = 0.0014720820228094722
iif = -0.0021448018041115902
ip = -0.01175184601377121
nb = -0.05616443581450401
for age in ages:
    print(age)
    print(age + age * em)
    print(age + age * iif)
    print(age + age * ip)
    print(age + age * nb)

# Em
ages = [1.97, 3.2, 3.25, 4.1]
im2 = -0.0014742523332587172
iif = -0.0035416196528567333
ip = -0.011683582098749017
nb = -0.06303181911515632
for age in ages:
    print(age)
    print(age + age * im2)
    print(age + age * iif)
    print(age + age * ip)
    print(age + age * nb)

# Iif
ages = [3.85, 4.15, 4.42, 4.54]
im2 = 0.002140209053085324
em = 0.0035291164428186833
ip = -0.00914323556976113
nb = -0.05368552080102855
for age in ages:
    print(age)
    print(age + age * im2)
    print(age + age * em)
    print(age + age * ip)
    print(age + age * nb)

# Ip
ages = [3.79, 3.98, 4.29, 4.47, 4.56]
im2 = 0.011615339332036002
em = 0.011548647139337533
iif = 0.009060392706615798
nb = -0.04259937163136966
for age in ages:
    print(age)
    print(age + age * im2)
    print(age + age * em)
    print(age + age * iif)
    print(age + age * nb)


# Nb
ages = [3.84, 3.89, 4.12, 4.42]
im2 = 0.05317773640966127
em = 0.059294385852934894
iif = 0.05095019352230537
ip = 0.04085878727392898
for age in ages:
    print(age)
    print(age + age * im2)
    print(age + age * em)
    print(age + age * iif)
    print(age + age * ip)

# Equações das regiões
for reg in landing_regions:
    regions.remove(reg)

for i in range(0, len(regions)):
    print(i)
    reg = regions[i]
    print(reg)

for reg in regions:
    print(reg)
    craters_hist = craters[(craters['REGION'] == reg)]
    D = craters_hist['DIAM_CIRC_IMG'].value_counts(bins=1000, sort=False)
    N = D.index
    D = np.flipud(np.flipud(D).cumsum())
    I = []
    for k in range(len(N)):
        I.append(N[k].right)
    D = D/areas[reg]
    D = np.log10(D)
    I = np.log10(I)

    def get_equation(x, y):
        degree = 5
        coefs, res, _, _, _ = np.polyfit(x, y, degree, full=True)
        ffit = np.poly1d(coefs)
        print(ffit)
        return ffit
    eq = get_equation(I, D)

# Equações por região
import numpy as np
x = np.linspace(0.2, 1.2, 100)
y6 = 0.752*x**5 - 4.366*x**4 + 8.91*x**3 - 7.572*x**2 + 0.5263*x - 7.511  # Isc
y7 = -0.4609*x**5 + 1.931*x**4 - 2.768 * \
    x**3 + 1.699*x**2 - 2.403*x - 7.314  # Ic2
y8 = -0.7263*x**6 + 7.468*x**5 - 29.62*x**4 + \
    56.99*x**3 - 54.67*x**2 + 21.85*x - 10.48  # Ec
y9 = -0.2175*x**7 + 2.394*x**6 - 9.148*x**5 + 15.14 * \
    x**4 - 10.31*x**3 + 1.579*x**2 - 1.302*x - 7.312  # Iork
y10 = 9.772*x**7 - 64.5*x**6 + 169.9*x**5 - 227.2*x**4 + \
    161.8*x**3 - 58.64*x**2 + 7.552*x - 7.682  # Iiap
y11 = -3.343*x**8 + 34.17*x**7 - 143.7*x**6 + 321.6*x**5 - \
    413.5*x**4 + 307.5*x**3 - 126.1*x**2 + 23.2*x - 8.844  # Ib
y12 = -0.2926*x**7 + 2.873*x**6 - 10.74*x**5 + 19.09 * \
    x**4 - 16.6*x**3 + 6.995*x**2 - 3.936*x - 6.506  # Iohi
y13 = 0.4272*x**5 - 3.099*x**4 + 8.121*x**3 - 9.088*x**2 + 2.246*x - 7.795  # Ig
y14 = 2.038*x**7 - 17.07*x**6 + 57.04*x**5 - 97.28 * \
    x**4 + 89.55*x**3 - 42.96*x**2 + 7.48*x - 8.007  # Nbl
y15 = 1*x**7 - 7.714*x**6 + 23.32*x**5 - 35.07*x**4 + \
    27.44*x**3 - 10.75*x**2 + 0.07681*x - 7.419  # Cc
y16 = -2.301*x**7 + 18.97*x**6 - 62.44*x**5 + 104.4 * \
    x**4 - 93.19*x**3 + 42.84*x**2 - 10.9*x - 6.687  # INt
y17 = 0.5955*x**6 - 3.546*x**5 + 7.536*x**4 - \
    6.709*x**3 + 2.25*x**2 - 2.19*x - 7.17  # Iom
y18 = 0.7088*x**6 - 5.026*x**5 + 13.47*x**4 - \
    17.04*x**3 + 10.3*x**2 - 4.378*x - 7.108  # It
y19 = 2.406*x**7 - 18.5*x**6 + 56.36*x**5 - 86.85*x**4 + \
    71.58*x**3 - 30.58*x**2 + 4.047*x - 7.667  # pNb
y20 = -0.836*x**7 + 9.071*x**6 - 39.45*x**5 + 88.26 * \
    x**4 - 108.5*x**3 + 72.83*x**2 - 26.73*x - 3.976  # pNt
y21 = 0.4317*x**5 - 1.963*x**4 + 2.73*x**3 - 1.072*x**2 - 2.013*x - 7.239  # Im1
y22 = 35.68*x**6 - 61.97*x**5 + 16.9*x**4 + 22.59*x**3 - \
    16.67*x**2 + 1.937*x - 7.308  # Ecc ok (0 - 0,9)
y23 = -0.4572*x**6 + 4.346*x**5 - 15.73*x**4 + \
    27.03*x**3 - 22.32*x**2 + 5.761*x - 7.958  # Nt
y24 = -0.1329*x**5 + 1.263*x**4 - 4.45 * \
    x**3 + 7.808*x**2 - 9.063*x - 4.314  # Icc
y25 = 3.347*x**6 - 13.36*x**5 + 17.57*x**4 - 6.999 * \
    x**3 - 2.134*x**2 - 0.08596*x - 7.352  # Ibm
y26 = -2.048*x**7 + 11.33*x**6 - 22.26*x**5 + 17.72 * \
    x**4 - 3.239*x**3 - 2.565*x**2 - 0.4885*x - 7.468  # Itd
y27 = -3.875*x**7 + 25.37*x**6 - 65.67*x**5 + 85.19 * \
    x**4 - 57.95*x**3 + 19.85*x**2 - 4.928*x - 7.32  # INp
y28 = -1.168*x**6 + 5.776*x**5 - 10.37*x**4 + \
    8.3*x**3 - 3.049*x**2 - 1.186*x - 7.365  # Id
y29 = 7.798*x**7 - 52.19*x**6 + 138.8*x**5 - 187.5*x**4 + \
    135.6*x**3 - 50.37*x**2 + 6.352*x - 7.774  # Iohs
y30 = -0.3933*x**5 + 3.505*x**4 - 11.47 * \
    x**3 + 17.06*x**2 - 13.27*x - 4.509  # Ioho
y31 = -0.2042*x**6 - 0.3976*x**5 + 5.004*x**4 - \
    10.45*x**3 + 8.245*x**2 - 4.057*x - 7.066  # Iia
y32 = 2.385*x**7 - 17.88*x**6 + 52.63*x**5 - 76.93 * \
    x**4 + 58*x**3 - 21.23*x**2 + 1.47*x - 7.32  # EIp
y33 = -3.47*x**5 + 15.44*x**4 - 23.23*x**3 + 13.59*x**2 - 4.581*x - 7.173  # Iic
y34 = 50.32*x**7 - 261.2*x**6 + 538*x**5 - 559.2*x**4 + 307.5 * \
    x**3 - 84.47*x**2 + 7.933*x - 7.648  # Ccc (0,2 - 1,5)
y35 = -0.5108*x**5 + 3.047*x**4 - 6.251 * \
    x**3 + 4.873*x**2 - 2.736*x - 7.409  # Nnj
y36 = -0.09113*x**7 + 0.4236*x**6 + 1.421*x**5 - 12.88 * \
    x**4 + 31.52*x**3 - 33.98*x**2 + 13.67*x - 9.351  # Nbsc
y37 = 1.451*x**6 - 9.64*x**5 + 23.84*x**4 - 27.3 * \
    x**3 + 15.09*x**2 - 5.814*x - 6.843  # Iorm
y38 = -0.2927*x**6 + 1.904*x**5 - 4.043*x**4 + \
    2.517*x**3 + 1.678*x**2 - 3.915*x - 6.93  # Icf
y39 = -0.1908*x**5 + 1.411*x**4 - 3.566 * \
    x**3 + 3.851*x**2 - 3.61*x - 6.995  # Ic
y40 = -2.704*x**6 + 17.32*x**5 - 39.41*x**4 + 39.77*x**3 - \
    18.19*x**2 + 1.576*x - 7.539  # Csc ok (02, - 1,5)
y41 = -0.09688*x**6 + 0.979*x**5 - 3.555*x**4 + \
    5.574*x**3 - 3.213*x**2 - 2.19*x - 6.763  # pNc
y42 = 1.01*x**5 - 4.975*x**4 + 8.507*x**3 - 5.926*x**2 - 0.4201*x - 7.79  # Np
y43 = 0.3826*x**5 - 2.211*x**4 + 4.644*x**3 - \
    4.364*x**2 - 0.1666*x - 7.371  # pNbm
y44 = 72.49*x**6 - 227.2*x**5 + 267.4*x**4 - 145.2*x**3 + \
    35.57*x**2 - 5.335*x - 7.184  # Imd ok (0.2 - 1.1)
y45 = -0.116*x**5 + 0.9179*x**4 - 2.771 * \
    x**3 + 3.962*x**2 - 4.516*x - 6.658  # Nc
y46 = 0.06201*x**6 + 0.1778*x**5 - 3.133*x**4 + \
    9.396*x**3 - 10.92*x**2 + 2.925*x - 7.656  # Ntp
y47 = 0.1674*x**5 - 1.814*x**4 + 5.463 * \
    x**3 - 6.325*x**2 + 0.9408*x - 7.719  # Ic1
y48 = -0.2705*x**7 + 2.53*x**6 - 9.215*x**5 + 16.55 * \
    x**4 - 15.08*x**3 + 6.164*x**2 - 2.59*x - 7.081  # Nbm
y49 = 1.172*x**5 - 7.609*x**4 + 17.79*x**3 - 17.73*x**2 + 4.96*x - 7.888  # Esc

craters_hist = craters[(craters['REGION'] == 'Imd')]
D = craters_hist['DIAM_CIRC_IMG'].value_counts(bins=1000, sort=False)
N = D.index
D = np.flipud(np.flipud(D).cumsum())
I = []
for k in range(len(N)):
    I.append(N[k].right)
D = D/areas['Ecc']
D = np.log10(D)
I = np.log10(I)

plt.plot(I, D)
plt.plot(x, y22)


def get_equation(x, y, d):
    degree = d
    coefs, res, _, _, _ = np.polyfit(x, y, degree, full=True)
    ffit = np.poly1d(coefs)
    print(ffit)
    return ffit


eq = get_equation(I, D, 7)

y1 = -0.33*x**5 + 3.028*x**4 - 10.13*x**3 + 14.98*x**2 - 11.33*x - 5.178  # Im2
y2 = -0.3179*x**5 + 2.796*x**4 - 8.945*x**3 + 12.74*x**2 - 9.7*x - 5.524  # Em
y3 = 1.003*x**5 - 6.561*x**4 + 15.55*x**3 - 16.14*x**2 + 5.023*x - 7.961  # Iif
y4 = 0.5839*x**5 - 4.062*x**4 + 10.15*x**3 - 10.99*x**2 + 2.87*x - 7.718  # Ip
y5 = 0.9198*x**5 - 5.209*x**4 + 10.3*x**3 - 8.25*x**2 + 0.4312*x - 7.686  # Nb

q = Im2
q = Em
q = Iif
q = Ip
q = Nb

Im2_values = []
Em_values = []
Iif_values = []
Ip_values = []
Nb_values = []

for j in range(6, 50):
    if j == 6:        w = y6
    if j == 7:        w = y7
    if j == 8:        w = y8
    if j == 9:        w = y9
    if j == 10:        w = y10
    if j == 11:        w = y11
    if j == 12:        w = y12
    if j == 13:        w = y13
    if j == 14:        w = y14
    if j == 15:        w = y15
    if j == 16:        w = y16
    if j == 17:        w = y17
    if j == 18:        w = y18
    if j == 19:        w = y19

    if j == 20:        w = y20
    if j == 21:        w = y21
    if j == 22:        w = y22
    if j == 23:        w = y23
    if j == 24:        w = y24
    if j == 25:        w = y25
    if j == 26:        w = y26
    if j == 27:        w = y27
    if j == 28:        w = y28
    if j == 29:        w = y29

    if j == 30:        w = y30
    if j == 31:        w = y31
    if j == 32:        w = y32
    if j == 33:        w = y33
    if j == 34:        w = y34
    if j == 35:        w = y35
    if j == 36:        w = y36
    if j == 37:        w = y37
    if j == 38:        w = y38
    if j == 39:        w = y39

    if j == 40:        w = y40
    if j == 41:        w = y41
    if j == 42:        w = y42
    if j == 43:        w = y43
    if j == 44:        w = y44
    if j == 45:        w = y45
    if j == 46:        w = y46
    if j == 47:        w = y47
    if j == 48:        w = y48
    if j == 49:        w = y49

    y = q-w
    r = y
    for i in range(0, len(y)):
        r[i] = y[i]/q[i]
    print(r.mean())
    print(np.median(r))
    #Im2_values.append(np.median(r))
    #Em_values.append(np.median(r))
    #Iif_values.append(np.median(r))
    #Ip_values.append(np.median(r))
    Nb_values.append(np.median(r))
    print()

# Idades
age_Im2 = 3.9 #+- 0.03
age_Im2_m = 3.87
age_Im2_M = 3.93
for rate in Im2_values:
    print(age_Im2 + age_Im2 * rate)
    print(round(((age_Im2_M + age_Im2_M * rate) - (age_Im2_m + age_Im2_m * rate))/2, 2))

age_Em = 4.1 #+- 0.01
age_Em_m = 4.09
age_Em_M = 4.11
for rate in Em_values:
    print(age_Em + age_Em * rate)
    print(round(((age_Em_M + age_Em_M * rate) - (age_Em_m + age_Em_m * rate))/2, 2))

age_Iif = 4.15 #+- 0.1
age_Iif_m = 4.05
age_Iif_M = 4.25
for rate in Iif_values:
    print(age_Iif + age_Iif * rate)
    print(round(((age_Iif_M + age_Iif_M * rate) - (age_Iif_m + age_Iif_m * rate))/2,1))

age_Ip = 3.98 #+- 0.01
age_Ip_m = 3.97
age_Ip_M = 3.99
for rate in Ip_values:
    print(age_Ip + age_Ip * rate)
    print(round(((age_Ip_M + age_Ip_M * rate) - (age_Ip_m + age_Ip_m * rate))/2, 2))

age_Nb = 3.84 #+- 0.04
age_Nb_m = 3.80
age_Nb_M = 3.88
for rate in Nb_values:
    print(age_Nb + age_Nb * rate)
    print(round(((age_Nb_M + age_Nb_M * rate) - (age_Nb_m + age_Nb_m * rate))/2, 2))

ages_final = []
for i in range(0, len(Im2_values)):
    age_im2 = age_Im2 + age_Im2 * Im2_values[i]
    age_em = age_Em + age_Em * Em_values[i]
    age_iif = age_Iif + age_Iif * Iif_values[i]
    age_ip = age_Ip + age_Ip * Ip_values[i]
    age_nb = age_Nb + age_Nb * Nb_values[i]
    ages_final.append(round((age_im2 + age_em + age_iif + age_ip + age_nb)/5, 2))
    

uncertainty_M = []
for i in range(0, len(Im2_values)):
    age_im2 = age_Im2_M + age_Im2_M * Im2_values[i]
    age_em = age_Em_M + age_Em_M * Em_values[i]
    age_iif = age_Iif_M + age_Iif_M * Iif_values[i]
    age_ip = age_Ip_M + age_Ip_M * Ip_values[i]
    age_nb = age_Nb_M + age_Nb_M * Nb_values[i]
    uncertainty_M.append((age_im2 + age_em + age_iif + age_ip + age_nb)/5)
    

uncertainty_m = []
for i in range(0, len(Im2_values)):
    age_im2 = age_Im2_m + age_Im2_m * Im2_values[i]
    age_em = age_Em_m + age_Em_m * Em_values[i]
    age_iif = age_Iif_m + age_Iif_m * Iif_values[i]
    age_ip = age_Ip_m + age_Ip_m * Ip_values[i]
    age_nb = age_Nb_m + age_Nb_m * Nb_values[i]
    uncertainty_m.append((age_im2 + age_em + age_iif + age_ip + age_nb)/5)
    

uncertainty = []
for i in range(0, len(Im2_values)):
    uncertainty.append(round((uncertainty_M[i] - uncertainty_m[i])/2, 2))

######################################################################################
# Plotando Crateras por clusters
craters_cluster = craters[(craters['CLUSTER'] != 0)]
craters_cluster = craters_cluster[(craters_cluster['CLUSTER'] <= 10)]
print(craters_cluster)

craters_cluster['NLON_CIRC_IMG'] = craters_cluster.apply(
    lambda row: row.LON_CIRC_IMG * 2.8444, axis=1)
craters_cluster['NLAT_CIRC_IMG'] = craters_cluster.apply(
    lambda row: (row.LAT_CIRC_IMG - 90) * -2.8444, axis=1)

# Plotando o Mapa
map = moon_map.MoonMapSmall()
plt.imshow(map.image)

plt.scatter(x=craters_cluster['NLON_CIRC_IMG'], y=craters_cluster['NLAT_CIRC_IMG'],
            c=craters_cluster['CLUSTER'], s=1)  # label=craters_cluster['CLUSTER'])
plt.show()

######################################################################################

######################################################################################

######################################################################################

######################################################################################