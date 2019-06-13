import xml.etree.ElementTree as ET
import os
import os.path
import pickle
import re


def convert(labels_ID):
    status = ''
    if not labels_ID:
        status = 'train'
        train = True
        number_of_labels = 0
    else:
        status = 'test'
        train = False

    base_dir = './resources/wipo-alpha/' + status

    if not os.path.isdir(base_dir):
        print("Dataset not found")
        print("please download wipo-alpha dataset to resources folder")
        exit()

    all_files_text = './resources/' + status + 'Directory.txt'
    cutoff = 2500

    texts = []
    labels = []
    all_files = open(all_files_text, 'r')
    for entry in all_files:
        path = base_dir + entry[1:]
        path = path.replace('\n', '')
        try:
            tree = ET.parse(path)
        except:
            print("File " + path + " not found")
            continue
        root = tree.getroot()
        text = ''
        found_desc = 0
        text_claims = ''  # Claims should be after description
        for child in root:
            if child.attrib.get('mc') is not None:
                label = child.attrib['mc'][:4]
            for child_child in child:
                if child_child.tag in ['ti']:
                    text += child_child.text + ' '
                if child_child.tag in ['ab', 'txt']:
                    if child_child.tag in ['txt']:
                        found_desc = 1
                    try:
                        text += child_child.text + ' '
                    except:
                        pass
                if child_child.tag in ['cl']:
                    try:
                        text_claims += child_child.text + ' '
                    except:
                        pass
                if text_claims is not '':
                    if found_desc:
                        found_desc = 0
                        text += text_claims

        text = text.replace('\n', ' ')
        text = re.sub('<[^>]+>', '', text)  # remove tags
        regex = re.compile('[^a-zA-Z]')  # remove all but alphabetic
        text = regex.sub(' ', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)  # remove words of length < 3
        text = " ".join(text.split()[:cutoff]).lower()

        if train:
            if label not in labels_ID:
                labels_ID[label] = number_of_labels
                number_of_labels += 1
        if label not in labels_ID:
            continue
        id_of_label = labels_ID.get(label)
        texts.append(text)
        labels.append(id_of_label)
    save_data(status, texts, labels, labels_ID)


def save_data(status, texts, labels, labels_ID):
    with open('./resources/' + status + '_texts.pkl', 'wb') as pckl:
        pickle.dump(texts, pckl)
    with open('./resources/' + status + '_labels.pkl', 'wb') as pckl:
        pickle.dump(labels, pckl)
    if status is 'train':
        with open('./resources/labels_ID.pkl', 'wb') as pckl:
            pickle.dump(labels_ID, pckl)
    print('Done! Number of classes of ' + status + ' documents is ' +
          str(len(labels_ID)))
    print("total number of samples is " + str(len(texts)))
