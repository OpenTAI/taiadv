from common.util import *
from setup_paths import *

class CIFAR10CNN:
    def __init__(self, mode='train', filename="cnn_cifar10.pt", epochs=100, batch_size=512):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.min_pixel_value, self.max_pixel_value = load_cifar10()
        # Swap axes to PyTorch's NCHW format
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2)).astype(np.float32)
        self.x_test = np.transpose(self.x_test, (0, 3, 1, 2)).astype(np.float32)
        self.input_shape = self.x_train.shape[1:]
        print(self.input_shape)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.x_train = transform_train(torch.from_numpy(self.x_train)).numpy()
        self.x_test = transform_test(torch.from_numpy(self.x_test)).numpy()

        if mode=='train':
            # build model
            self.classifier = torchvision.models.resnet18(weights='DEFAULT')
            fc_features = self.classifier.fc.in_features 
            self.classifier.fc = nn.Linear(in_features=fc_features, out_features=self.num_classes)
            self.classifier = self.art_classifier(self.classifier)
            # train model
            self.classifier.fit(self.x_train, self.y_train, batch_size=self.batch_size, nb_epochs=self.epochs)
            # save model
            torch.save(self.classifier.model, str(os.path.join(checkpoints_dir, self.filename)))
        elif mode=='load':
            self.classifier = self.art_classifier(torch.load(str(os.path.join(checkpoints_dir, self.filename))))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
        
        pred = self.classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("mode option: {}. Accuracy on benign test examples: {}%".format(self.mode, accuracy * 100))
    
    def art_classifier(self, net):
        net.to(self.device)
        # summary(net, input_size=self.input_shape)
        # print(net)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(self.min_pixel_value, self.max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=self.input_shape,
            nb_classes=self.num_classes,
        )
        
        return classifier