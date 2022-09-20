import os


# paths
class Paths():
    def __init__(self):
        # root
        self.root = os.path.abspath(os.path.dirname(__file__))

        # data
        self.data = os.path.join(self.root, 'data')
        self.air_data = os.path.join(self.data, 'AIR-Act2Act')
        self.ntu_data = os.path.join(self.data, 'NTU')

        self.extracted = os.path.join(self.root, 'extracted')
        self.air_extracted = os.path.join(self.extracted, 'AIR-Act2Act')
        self.ntu_extracted = os.path.join(self.extracted, 'NTU')

        # model
        self.model = os.path.join(self.root, 'models')
        self.kmeans_model = os.path.join(self.model, 'k-means')
        self.lstm_model = os.path.join(self.model, 'lstm')

path = Paths()


# hyper parameters
class HParams():
    def __init__(self):
        # neural model parameters
        self.use_cuda = True
        self.lstm_hidden_size = 256
        self.input_norm_method = 'vector'  # ['vector', 'torso']
        self.user_pose_length = 15
        self.user_pose_dim = 3 * 8 + 1
        self.hold_last = 5

        # training
        self.epochs = 300
        self.save_epochs = 10
        self.batch_size = 50
        self.learning_rate = 0.00001
        self.step = 3
        self.b_use_noise = True

        # robot test
        self.v_width = 640
        self.v_height = 480
        self.actions = ['A005']
        # self.actions = ['A001', 'A004', 'A005', 'A006', 'A008']

hp = HParams()
