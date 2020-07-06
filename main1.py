'''
Description: Main file to predict the face mask detection by training a new model
Author: Prashant
Team no: 5
Project no: 5
Command to run this file: python main.py -models/deploy.prototxt -cm models/deploy.caffemodel -m models/model.model
'''


from classes.faceMask import FaceMask
import argparse

ap = argparse.ArgumentParser()
ap.addArgument("-p", "--prototxt", type = str, help = "Path to prototxt file")
ap.addArgument("-ds", "--dataset", type = str, help = "Path to training dataset")
ap.addArgument("-cm", "--caffemodel", type = str, help = "Path to caffeModel file")
ap.addArgument("-m", "--model", type = str, help = "Path to mask detection model")
ap.addArgument("-c", "--confidence", default = 0.5, type = float, help = " min confidence for filtering")
args = vars(ap.parse_args())

if __name__ == __main__ :
    fm = FaceMask(path_to_dataset = args["dataset"], path_to_prototxt = args["prototxt"], path_to_caffe = args["caffemodel"], path_to_model = args["model"], confidence = args["confidence"])

    d, t = [], []
    d, t = fm.prepData()
    fm.creatArchitecture()
    fm.trainModel(data = d,target = t)
    fm.detectMask()
