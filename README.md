# ecosystem project pipeline

## Data
- Land use map: https://www.sciencebase.gov/catalog/item/664e0d2bd34e702fe8744536
- Satellite image: Dr. Zhao hipergator

## Code
### simple demo
- truth_branford.py: way to generate label
- run.py: a simple case to demonstrate how the pipeline works
  
### temporal
- preprocess_map.py: extract Polk County by shapefile from the whole land use map and random sampling points
- preprocess_label.py: preprocess the labels and split into training and validation
- main.py: run the pipeline to train a model and use the trained model to predict land use/ecosystem service map

### spatial(soon)

## paper
https://ieeexplore.ieee.org/abstract/document/10642804

