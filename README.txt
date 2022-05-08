# CheckDetection_and_ProjectionCorrection

Attached is a yaml file with the conda environment that I used to run this project. I have poor project controlling standards, so there are many packages in the conda environment that are not used in this particular project at all, I apologize.

This project also utilizes a segmentation model named detectron2 and it is run with pyyaml 5.1, they must be pip installed with these commands before the project is run:
	!pip install pyyaml==5.1
	!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html

Files:
	proj3.py -- Main body of code
	proj3Helper.py -- Helper file containing functions to be used
	blankcheck2.jpg -- file containing a template used for pattern matching. Must be accessible by the code
	samples/ -- file that holds images to be tested on

How to run, 2 arguments:
	input_folder: the folder that holds the images to be run
	path_to_template: the path to the template image I used

example run: python proj3.py --input_folder ImagesToTestOn --path_to_template template/blankcheck2.jpg

(I notice for some reason it quickly exits the final picture of a sequence when running it on a folder, if that happens just put the final picture in a new folder, I apologize.)
	