# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path
from typing import List

import sys
from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight,
                                  guess_model_task)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, callbacks, yaml_load
from ultralytics.yolo.utils.checks import check_yaml, check_imgsz
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
MODEL_MAP = {
    "classify": [
        ClassificationModel, 'yolo.TYPE.classify.ClassificationTrainer', 'yolo.TYPE.classify.ClassificationValidator',
        'yolo.TYPE.classify.ClassificationPredictor'],
    "detect": [
        DetectionModel, 'yolo.TYPE.detect.DetectionTrainer', 'yolo.TYPE.detect.DetectionValidator',
        'yolo.TYPE.detect.DetectionPredictor'],
    "segment": [
        SegmentationModel, 'yolo.TYPE.segment.SegmentationTrainer', 'yolo.TYPE.segment.SegmentationValidator',
        'yolo.TYPE.segment.SegmentationPredictor']}


class YOLO:
    """
    YOLO

    A python interface which emulates a model-like behaviour by wrapping trainers.
    """

    def __init__(self, model='yolov8n.pt', type="v8") -> None:
        """
        Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
        """
        
        # version of model to use 
        self.type = type
        
        # classes used
        self.ModelClass = None  # model class
        self.TrainerClass = None  # trainer class
        self.ValidatorClass = None  # validator class
        self.PredictorClass = None  # predictor class
        
        # type of predictor to use 
        self.predictor = None  # reuse predictor
        
        # model and trainer objects to be used 
        self.model = None  # model object
        self.trainer = None  # trainer object
        
        # task type [ Detection, Segmentation, Classification ]
        self.task = None  # task type
        
        #ckpt means checkpoint pytorch & .py is pytorch file
        self.ckpt = None  # if loaded from *.pt
        
        # configuration file // .yaml format
        self.cfg = None  # if loaded from *.yaml
        
        # path to the latest checkpoint
        self.ckpt_path = None
        
        # overrides for the trainer object 
        self.overrides = {}  # overrides for trainer object
        
        

        # Load or create new YOLO model
        
        # creates a dictionary names load_methods
        # inside the dictionary are two key value pairs 
        
        # the first key is '.pt' which is the pytorch file ending 
        # this key maps to self._load which is not currently defined, but it is a method inside this class
        
        # the second key is '.yaml' which is the configuration file ending 
        # this key maps to self._new which is not currently defined, but is a method inside this class
        load_methods = {'.pt': self._load, '.yaml': self._new}
        
        # makes the model string a path object, then extracts the suffix (the file type) from it
        # stored in the suffix variable (string)
        suffix = Path(model).suffix
        
        
        # depending on the suffix of the path/str stored in model, different methods are called
        
        # if the model ends in a pytorch file type ('.pt'): ._load(model) is called on the YOLO object - this creates a new model
        # if the model ends in a yaml file: ._new(model) is called on the YOLO object - this creates a new model
    
        if suffix in load_methods:
            {'.pt': self._load, '.yaml': self._new}[suffix](model)
        else:
            raise NotImplementedError(f"'{suffix}' model loading not implemented")
            
            

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)
    
    

    def _new(self, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            verbose (bool): display model info on load
        """
        cfg = check_yaml(cfg)  # check YAML - this returns a file path in a string format - ended here 2/14
        
        ## yaml_load - Load YAML data from a file. RETURNS: dict: YAML data and file name.
        cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
        
        # guess model task - Guess the task of a PyTorch model from its architecture or configuration.
        # takes in model (nn.Module) or (dict): PyTorch model or model configuration in YAML format. CONFIGURATION FILE IN THIS CASE
        # RETURNS str: Task of the model ('detect', 'segment', 'classify').
        # SyntaxError: If the task of the model could not be determined.
        self.task = guess_model_task(cfg_dict) #self.task is a string, in this case it is 'detect'
        
        #_assign_ops_from_task should be in this class, sends in self.task which is 'detect' in this case
        # _assign_ops_from_task returns the classes to use fpr training, validating and predicting
        
        # self.ModelClass      -> ClassificationModel
        # self.TrainerClass    -> yolo.v8.classify.ClassificationTrainer
        # self.ValidatorClass  -> yolo.v8.classify.ClassificationValidator
        # self.PredictorClass  -> yolo.v8.classify.ClassificationPredictor
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)
        
        # here we initialize the model 
        # self.ModelClass(cfg_dict, verbose=verbose) is equal to ClassificationModel(cfg_dict,verbose=verbose)
        # this call returns an object of ClassificationModel class
        # ultralytics.nn.tasks.ClassificationModel(cfg_dict,verbose=verbose)
        self.model = self.ModelClass(cfg_dict, verbose=verbose)  # initialize w parameters  # yaml, model, channels, number of classes, cutoff index, verbose flag
        
        #saves off the configuration file 
        self.cfg = cfg

    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
        """
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.ckpt_path = weights
        self.task = self.model.args["task"]
        self.overrides = self.model.args
        self._reset_ckpt_args(self.overrides)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._assign_ops_from_task(self.task)

    def reset(self):
        """
        Resets the model modules.
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose=False):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)

    def fuse(self):
        self.model.fuse()

    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        overrides["save"] = kwargs.get("save", False)  # not save files by default
        if not self.predictor:
            self.predictor = self.PredictorClass(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        is_cli = sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides["rect"] = True  # rect batches as default
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = self.ValidatorClass(args=args)
        validator(model=self.model)
        return validator.metrics

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed

        exporter = Exporter(overrides=args)
        exporter(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        # makes a copy of the overrides dictionary 
        overrides = self.overrides.copy()
        # update is performed on dictionary, and it is sent in a dictionary
        # if the key already exists, it replaces the value, else it adds a key and value 
        overrides.update(kwargs)
        
        # if the keyword arguments have a new cfg file, update logger, update cfg 
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]), append_filename=True)
            
        # adding the task and mode to overrides dictionary 
        overrides["task"] = self.task
        overrides["mode"] = "train"
        
        # if no data was passed, raise an error
        if not overrides.get("data"):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        
        # if overrides contains a resume key, this means we want to pick up from an old checkpoint
        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path
            
        
        # any arguments in the overrides file are integrated into the configuration file
        self.trainer = self.TrainerClass(overrides=overrides) # updates the trainer with new configuration file with overrides
        
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        
        # self.trainer points to a BaseTrainer object found in ultralytics/yolo/engine/trainer.py
        # this training call calls the BaseTrainer train method ## ended here 2/15
        
        self.trainer.train()
        # update model and cfg after training
        if RANK in {0, -1}:
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self.model.to(device)

    def _assign_ops_from_task(self, task):
        #"classify": [
        # ClassificationModel, 
        # 'yolo.TYPE.classify.ClassificationTrainer', 
        # 'yolo.TYPE.classify.ClassificationValidator',
        # 'yolo.TYPE.classify.ClassificationPredictor'
        #]
        
        model_class, train_lit, val_lit, pred_lit = MODEL_MAP[task] 
        
        # warning: eval is unsafe. Use with caution
        # replace TYPE with version number
        # example: 'yolo.TYPE.classify.ClassificationTrainer' -> 'yolo.v8.classify.ClassificationTrainer'
        # then it performs eval() on it which will actually call whatever is inside the method call
        # for example it will call yolo.v8.classify.ClassificationTrainer
        # So, it is saving the correct classes in the class variables, then returning them for use 
        trainer_class = eval(train_lit.replace("TYPE", f"{self.type}"))
        validator_class = eval(val_lit.replace("TYPE", f"{self.type}"))
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))
        
        # so now these variables are pointing to these classes:
        # trainer_class -> yolo.v8.classify.ClassificationTrainer
        # validator_class -> yolo.v8.classify.ClassificationValidator
        # predictor_class  -> yolo.v8.classify.ClassificationPredictor

        return model_class, trainer_class, validator_class, predictor_class

    @property
    def names(self):
        """
         Returns class names of the loaded model.
        """
        return self.model.names

    @property
    def transforms(self):
        """
         Returns transform of the loaded model.
        """
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    @staticmethod
    def add_callback(event: str, func):
        """
        Add callback
        """
        callbacks.default_callbacks[event].append(func)

    @staticmethod
    def _reset_ckpt_args(args):
        for arg in 'augment', 'verbose', 'project', 'name', 'exist_ok', 'resume', 'batch', 'epochs', 'cache', \
                'save_json', 'half', 'v5loader', 'device', 'cfg', 'save', 'rect', 'plots':
            args.pop(arg, None)
