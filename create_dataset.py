from convertXMLtoPCKL import convert
import argparse
import os.path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-train', action="store_true")
parser.add_argument('-test', action="store_true")

args = vars(parser.parse_args())

if (args['train'] and args['test']) or (not args['train'] and not args['test']):
    print("Error in arguments")
    print("run \"python create_data.py\" with -train or -test argument")
    exit()

if args['train']:
    LabelsIdDict = {}
    convert(LabelsIdDict)
else:
    labels_ID_path = './resources/labels_ID.pkl'
    if os.path.isfile(labels_ID_path):
        with open('./resources/labels_ID.pkl', 'rb') as pckl:
            LabelsIdDict = pickle.load(pckl)
        convert(LabelsIdDict)
    else:
        print("run code with \"-train\" argument " +
              "first to create label alphabet")
        exit()
