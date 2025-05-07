"""
For:
Author: zhihu
Date: 2023/01/12
"""
"""
For:
Author: zhihu
Date: 2023/01/12
"""

import numpy as np
import geopandas as gpd
import pandas as pd

gdf=gpd.read_file(r"C:\Users\zhihui.tian\Desktop\eco_more_exp\trainingfrom_branford\trainingfrom_branford.shp")
gdf= gdf.loc[(gdf["SAMPLE_1"] != '`7') & (gdf["SAMPLE_1"] != "9'") & (gdf["SAMPLE_1"] != np.nan)& (gdf["SAMPLE_1"] != np.nan)]
gdf=gdf.dropna()
class_names=gdf['SAMPLE_1'].unique()
class_names.sort()
print('class names',class_names)
class_ids=np.arange(class_names.size)+1
print('class ids',class_ids)
df=pd.DataFrame({'labels':class_names,'id':class_ids})
df.to_csv("C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/class_lookup.csv")
print("gdf without ids",gdf.head())
gdf['id']=gdf['SAMPLE_1'].map(dict(zip(class_names,class_ids)))
print('gdf with ids',gdf.head())

gdf_train=gdf.sample(frac=0.8)
gdf_test=gdf.drop(gdf_train.index)
print('gdf shape',gdf.shape,'training shape',gdf_train.shape,'test',gdf_test.shape)
gdf_train.to_file("C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/train.shp")
gdf_test.to_file("C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/test.shp")