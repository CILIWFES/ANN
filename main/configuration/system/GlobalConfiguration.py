import configparser
from .GlobalConstant import GlobalConstant

GLOCT = GlobalConstant()

class GlobalConfiguration:

    def __init__(self):
        self.__GConfig = configparser.ConfigParser()
        self.__GConfig.read(GLOCT.GLO_CONFIG_PATH)

    def getConfig(self, section, name):
        return self.__GConfig.get(section, name)
