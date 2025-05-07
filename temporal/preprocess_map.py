import os
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point
import numpy as np
from rasterio.plot import show
from rasterio.mask import mask
import random


shapefile_path = r'C:\Users\zhihui\Desktop\eco_pro\polk_county.shp'
input_dir = r"C:\Users\zhihui\Desktop\eco_pro"
output_dir = r"C:\Users\zhihui\Desktop\eco_pro\extracted"
os.makedirs(output_dir, exist_ok=True)
################################# --- extract polk county land use map based on .shp file(get the label for training)--- #################################
# --- Load the shapefile ---
gdf = gpd.read_file(shapefile_path)

# --- Process each year ---
for year in range(2008, 2024):  # includes 2023
    filename = f"Annual_NLCD_LndCov_{year}_CU_C1V0.tif"
    input_path = os.path.join(input_dir, filename)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    with rasterio.open(input_path) as src:
        # Reproject shapefile to match raster CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Mask the raster with the shapefile geometry
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()

    # Update metadata for the new clipped raster
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Save the extracted file
    output_path = os.path.join(output_dir, f"land_use_extracted_{year}.tif")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Saved extracted raster for {year} to {output_path}")


################################# --- sampling points from the land use map(get the label for training)--- #################################
shapefile_path = r'C:\Users\zhihui\Desktop\eco_pro\polk_county.shp'
extracted_dir = r"C:\Users\zhihui\Desktop\eco_pro\extracted"
output_points_dir = r"C:\Users\zhihui\Desktop\eco_pro\sampled_points"
os.makedirs(output_points_dir, exist_ok=True)

# Load the shapefile (for geometry and CRS)
gdf = gpd.read_file(shapefile_path)
geometry = gdf.unary_union  # Combine all shapes into one (if multiple)
crs = gdf.crs

# --- Function: Generate random points inside a polygon ---
def generate_random_points(polygon, num_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < num_points:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

# --- Main loop for each year ---
for year in range(2008, 2024):
    tif_path = os.path.join(extracted_dir, f"land_use_extracted_{year}.tif")
    if not os.path.exists(tif_path):
        print(f"File not found: {tif_path}")
        continue

    with rasterio.open(tif_path) as src:
        # Reproject polygon to match raster CRS
        if gdf.crs != src.crs:
            gdf_proj = gdf.to_crs(src.crs)
            geom_proj = gdf_proj.unary_union
        else:
            geom_proj = geometry

        # Generate random points inside polygon
        random_points = generate_random_points(geom_proj, num_points=5000)  # Adjust number as needed

        # Get raster values at those points
        coords = [(p.x, p.y) for p in random_points]
        values = list(src.sample(coords))
        values = [v[0] if v[0] is not None else -9999 for v in values]  # Fallback value

        # Build GeoDataFrame of sampled points
        points_gdf = gpd.GeoDataFrame({
            'year': [year] * len(coords),
            'land_use': values
        }, geometry=random_points, crs=src.crs)

        # Save to file
        output_path = os.path.join(output_points_dir, f"sampled_points_{year}.shp")
        points_gdf.to_file(output_path)
        print(f"Saved sampled points for {year} to {output_path}")
