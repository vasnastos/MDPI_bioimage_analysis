from models import ImageNet

def scenario1():
    imageNN=ImageNet()
    imageNN.extract_features(base_model_name='vgg16',lb=0,ub=10,save=True)


if __name__=='__main__':
    scenario1() # Extracted image features