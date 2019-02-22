import configparser
from .GlobalConstant import GlobalConstant

GLOCT = GlobalConstant()


class GlobalConfiguration:

    def __init__(self):
        self.__GConfig = configparser.ConfigParser()
        self.__GConfig.read(GLOCT.SYS_GLO_CONFIG_PATH)

    def getConfig(self, section, name):
        return self.__GConfig.get(section, name)

    def getFilsPath(self, section, names:list):
        path = GLOCT.SYS_FILES_PATH
        for item in names:
            path += self.getConfig(section, item)
        return path
