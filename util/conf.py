from typing import Optional
from os import environ
from enum import Enum, unique

from ruamel.yaml import YAML

from util.log import get_logger


@unique
class Env(Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


ENV = Env(environ.get("ENV", Env.DEV))
print(f">> current env: {ENV.value}")
yaml = YAML()
logger = get_logger()
conf_dict = {}


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f_yaml:
        datas = yaml.load(f_yaml)
    return datas


def get_conf(conf_path: Optional[str] = None) -> dict:
    if conf_path is None:
        if ENV == Env.DEV:
            conf_path = "./conf/default.yml"
        else:
            conf_path = f"./conf/default.{ENV.value}.yml"

    if conf_path not in conf_dict.keys():
        conf_dict[conf_path] = read_yaml(conf_path)
        logger.info(f"current env: {ENV.value}")
    return conf_dict[conf_path]
