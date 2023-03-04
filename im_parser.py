import os,json

class JSON_Parser:
    path_to_data_files = os.path.join('..','Dataset')

    def __init__(self):
        self._training_json_file = os.path.join(JSON_Parser._path_to_data_files, 'Extracted Annotation Data', 'training_set.json')
        self._validation_json_file = os.path.join(JSON_Parser._path_to_data_files, 'Extracted Annotation Data', 'validation_set.json')

    def load_json_files(self):
        with open(self._training_json_file, 'r') as training_data, open(self._validation_json_file, 'r') as validation_data:
                self._trainData = json.load(training_data)
                self._validationData = json.load(validation_data)
        return (self._trainData, self._validationData)

    def concatenate_json_data(self):
        trainData, validationData = self.load_json_files()
        self.allData = dict()
        self.allData = trainData | validationData
        return list(self.allData.values())