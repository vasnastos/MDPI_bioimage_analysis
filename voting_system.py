from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.
from models import ImageNet
from sklearn.model_selection import StratifiedKFold

class VotingSystem:
    def __init__(self):
        self.model=ImageNet()
        self.model.load(dit_biopsies=True)

        # set cross validation 
        self.cross_validation_model=StratifiedKFold(n_splits=10,shuffle=False)
    
    def vote(self):
        vote_model=VotingClassifier(
            estimators=[
                
            ]   
        )
