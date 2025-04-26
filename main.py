import numpy as np
import scipy
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, slic
import time
import scipy
from osgeo import gdal,ogr
from sklearn.ensemble import RandomForestClassifier
from pysnic.algorithms.snic import snic
from pysnic.algorithms.polygonize import polygonize
from pysnic.algorithms.ramerDouglasPeucker import RamerDouglasPeucker




"""multiple land use training"""
""""""
import pickle
naip_fn=r"D:\PlanetScope\8-Band\2023\March\Polk\composites\composite.tif"
driverTiff= gdal.GetDriverByName('GTiff')
naip_ds=gdal.Open(naip_fn)
segments = np.load(
    r"C:\Users\zhihui.tian\Downloads\image-analysis-20230131T194214Z-001\new_start\segments_polk.npy")
segment_ids = np.arange(5096)
objects = np.load(
    r"C:\Users\zhihui.tian\Downloads\image-analysis-20230131T194214Z-001\new_start\objects_polk.npy").tolist()
for yr in [2008,2013,2018,2023]:
    train_fn=rf"D:\polk_time\{yr}\train.shp"
    train_ds=ogr.Open(train_fn)
    lyr=train_ds.GetLayer()
    driver=gdal.GetDriverByName('MEM')
    target_ds=driver.Create('',naip_ds.RasterXSize, naip_ds.RasterYSize,1,gdal.GDT_UInt16)
    target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    target_ds.SetProjection(naip_ds.GetProjection())
    options=['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds,[1],lyr,options=options)
    ground_truth=target_ds.GetRasterBand(1).ReadAsArray()


    classes=np.unique(ground_truth)[1:]
    print('class values',classes)

    segments_per_class={}
    for klass in classes:
        segments_of_class=segments[ground_truth == klass]
        segments_per_class[klass]=set(segments_of_class)
        print("Training segments for class",klass,":",len(segments_of_class))

    intersection=set()
    accum=set()

    for class_segments in segments_per_class.values():
        intersection |=accum.intersection(class_segments)
        accum |=class_segments
    #assert len(intersection) ==0, "Segment(s) represent multiple classes"

    # 1- create training image
    train_img = np.copy(segments)
    # 2- need to treshold to identify maximum of segment value is
    threshold = train_img.max() + 1

    for klass in classes:
        class_label=threshold+klass
        for segment_id in segments_per_class[klass]:
            train_img[train_img==segment_id]=class_label

    train_img[train_img <= threshold] = 0
    train_img[train_img > threshold] -= threshold

    training_objects = []
    training_labels = []

    for k in classes:
        class_train_object = [value for i, value in enumerate(objects) if segment_ids[i] in segments_per_class[k]]
        # this code will show the repeat of class,
        # for example, if we had 15 segment represented water, we would then get number of 3 that repeated 15 times
        training_labels += [k] * len(class_train_object)
        # add training_objects
        training_objects += class_train_object
        print('Training objecs for class', k, ':', len(class_train_object))

    model = RandomForestClassifier(n_jobs=-1)

    import pandas as pd
    X = pd.DataFrame(training_objects)
    y = pd.Series(training_labels)

    # Drop rows where any feature is NaN
    X = X.dropna()
    y = y.loc[X.index]

    # Convert back if needed
    training_objects = X.values
    training_labels = y.values

    model.fit(training_objects, training_labels)
    print('Fitting Random Forest Classifier')

    with open(rf'D:\polk_time\{yr}\model_training_yearly300_changed.pkl', 'wb') as file:
        pickle.dump(model, file)


"""predicting"""

naip_fn=r"D:\PlanetScope\8-Band\2023\March\Polk\composites\composite.tif"
driverTiff= gdal.GetDriverByName('GTiff')
naip_ds=gdal.Open(naip_fn)
nbands=naip_ds.RasterCount
#
band_data=[]
print('bands',naip_ds.RasterCount,naip_ds.RasterYSize,'rows','columns',
      naip_ds.RasterXSize)
for i in range(1,nbands+1):
    band=naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data=np.dstack(band_data)
print(band_data.shape)

img=exposure.rescale_intensity(band_data)


for yr in [2008, 2013,2018,2023]:
    with open(rf"D:\polk_time\{yr}\model_training_yearly300_changed.pkl",'rb') as file:
        model=pickle.load(file)
    objects = np.load(r"C:\Users\zhihui.tian\Downloads\image-analysis-20230131T194214Z-001\new_start\objects_polk.npy").tolist()
    segments = np.load(r"C:\Users\zhihui.tian\Downloads\image-analysis-20230131T194214Z-001\new_start\segments_polk.npy")
    segment_ids = np.arange(5096)

    objects_test = objects
    # prediction on test_1
    predicted_scores = np.array([])
    for i in range(len(objects_test)):
        a = np.array(objects_test[i])
        a[np.isnan(a)] = 0

        predicted_score =np.argmax(model.predict_proba(np.array(a).reshape(-1, 1).T))
        predicted_scores = np.append(predicted_scores, predicted_score)

    print('Predicting Classifications')

    # copy of segments
    segments_test = np.copy(segments)
    clf = segments_test
    # predict segment_id

    segment_test_ids = segment_ids
    for segment_id, k in zip(segment_test_ids, predicted_scores):
        clf[clf == segment_id] = k

    print('Prediction applied to numpy array')

    # make a mask to show us where we have data and do not have data
    mask = np.sum(img, axis=2)
    mask[mask > 0.0] = 1.0
    mask[mask == 0.0] = -1.0
    clf = np.multiply(clf, mask)
    clf[clf < 0] = -9999.0
    # save and visualize classification data

    clf_ds = driverTiff.Create(
        rf"D:\polk_time\{yr}\5000sample_ocala.tif",
        naip_ds.RasterXSize, naip_ds.RasterYSize,
        1, gdal.GDT_Float32)
    clf_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    clf_ds.SetProjection(naip_ds.GetProjectionRef())
    clf_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clf_ds.GetRasterBand(1).WriteArray(clf)
    clf_ds = None


import numpy as np
import rasterio
mapping = np.array([11, 21, 22, 23, 24, 31, 42, 43, 52, 71, 81, 82, 90, 95])

for yr in [2008,2013,2018,2023]:

    input_path = rf"D:\polk_time\{yr}\5000sample_ocala.tif"
    output_path = rf"D:\polk_time\{yr}\5000sample_ocala_Vmap.tif"

    with rasterio.open(input_path) as src:
        data = src.read(1)  # Read first band
        profile = src.profile
    data = data.astype(np.int32)
    mapped_data = np.full_like(data, fill_value=-9999)  # or another nodata value you prefer

    # Only map valid data (0–13)
    valid_mask = (data >= 0) & (data < len(mapping))
    mapped_data[valid_mask] = mapping[data[valid_mask]]

    # Update the nodata value in the profile
    profile.update(dtype=rasterio.int32, nodata=-9999)

    # Save
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mapped_data, 1)
