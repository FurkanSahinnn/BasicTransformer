import yaml
from settings.settings import Model, Training, TestParameters, TestTexts

class AppConfig:
    model: Model
    training: Training
    test_parameters: TestParameters
    test_texts: TestTexts

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r",  encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
    
    def get_model(self):
        return self.config["model"]
    
    def get_training(self):
        return self.config["training"]
    
    def get_test_parameters(self):
        return self.config["test_parameters"]
    
    def get_test_texts(self):
        return self.config["test_texts"]

app_config = AppConfig()