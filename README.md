### Gaussian-mixture-based methodology to identify sunlit leaves in images.

Flir images should be placed in the data/flir_images/ directory.
Code that generates .csv files out of Flir images can be found in the code_for_csv_files/ directory.

After you generate the .csv files, place them in the data/csv_files folder
Make sure to adjust the prediction resolution and the output folder path in the GMM_Paper_csv.py script.
Run the GMM_Paper_csv.py script. Use the -act argument to run the GMM for all the csv_files.
You can use the -i argument to run the GMM for only one image.


After the predictions are generated, you can use the script found in results/generate_common_folder.py to generate the folder "Test_GMM" with _gt.png and _pred.png images (ground truths and predictions). This folder can then be used to generate confusion matrices and evaluation metrics. Again, remember to change the resolution of the images according to your needs.
