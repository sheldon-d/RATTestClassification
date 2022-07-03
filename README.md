# RATTestClassification
Experimental pipeline for classifiying COVID-19 RAT tests using image processing. Uses OpenCV with Python.

Note: This pipeline does not work with all images of RAT tests. It may require some further tuning of parameters for the different image processing steps to work with noisy images, images with reflective surfaces, and images with different resolutions. This is just a proof of concept showing that image processing techniques can be used to classify RAT tests and works well for the ten images provided.

## Running the pipeline for one image (Showing processing steps in detail)
### Command to run
```
python RATTestClassification.py images\<image-name>.png
```
### Example output figure
![OneTest](https://user-images.githubusercontent.com/57740952/177058471-54d3c0cc-3ea4-4bda-884f-7878b200c979.png)
### Example result
Result: Positive

## Running the pipeline for all images (Batch processing)
### Command to run
```
python RATTestClassification.py
```
### Example output figures
| Positive Tests  | Negative Tests |
| ------------- | ------------- |
| ![PositiveTests](https://user-images.githubusercontent.com/57740952/177058486-91c8ef5c-a76e-4db3-ab20-807aefafca99.png)  | ![NegativeTests](https://user-images.githubusercontent.com/57740952/177058495-8bf3d4c4-6dc5-4327-ace3-10c77114f743.png)  |
### Example results
1. Result: Positive
2. Result: Positive
3. Result: Positive
4. Result: Positive
5. Result: Positive
6. Result: Negative
7. Result: Negative
8. Result: Negative
9. Result: Negative
10. Result: Negative
