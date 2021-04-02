class BaseTrainer:
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer,
                 config,
                 num_attrs,
                 summary_writer,
                 num_classes,
                 attribute_list,
                 save_dir,
                 xi=.8,
                 with_attribute=False, **kwargs):
        self.model = model
        self.save_dir = save_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.num_attrs = num_attrs
        self.attribute_list = attribute_list
        self.summary_writer = summary_writer
        self.with_attribute = with_attribute
        assert num_classes != 0
        self.num_classes = num_classes
        self.xi = xi

    def train(self, epoch):
        raise NotImplementedError

    def eval(self, epoch):
        raise NotImplementedError
