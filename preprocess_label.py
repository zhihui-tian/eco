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
import os

for i in range(2008,2024):
    path = rf"C:\Users\zhihui\Desktop\image-analysis-20230131T194214Z-001\new_start\{i}"
    if not os.path.exists(path):
        os.makedirs(path)

    gdf=gpd.read_file(rf"C:\Users\zhihui\Desktop\eco_pro\sampled_points\sampled_points_{i}.shp")
    gdf= gdf.loc[(gdf['land_use'] != '`7') & (gdf['land_use'] != "9'") & (gdf['land_use'] != np.nan)& (gdf['land_use'] != np.nan)]
    gdf=gdf.dropna()
    class_names=gdf['land_use'].unique()
    class_names.sort()
    print('class names',class_names)
    class_ids=np.arange(class_names.size)+1
    print('class ids',class_ids)
    df=pd.DataFrame({'labels':class_names,'id':class_ids})
    df.to_csv(rf"C:\Users\zhihui\Desktop\image-analysis-20230131T194214Z-001\new_start\{i}\class_lookup.csv")
    print("gdf without ids",gdf.head())
    gdf['id']=gdf['land_use'].map(dict(zip(class_names,class_ids)))
    print('gdf with ids',gdf.head())

    gdf_train=gdf.sample(frac=0.8)
    gdf_test=gdf.drop(gdf_train.index)
    print('gdf shape',gdf.shape,'training shape',gdf_train.shape,'test',gdf_test.shape)
    gdf_train.to_file(rf"C:\Users\zhihui\Desktop\image-analysis-20230131T194214Z-001\new_start\{i}\train.shp")
    gdf_test.to_file(rf"C:\Users\zhihui\Desktop\image-analysis-20230131T194214Z-001\new_start\{i}\test.shp")