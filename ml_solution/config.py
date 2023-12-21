def _init():#初始化
    global config
    config = {}
 

def set(key, value):
    try:
        config[key] = value
        return True
    except KeyError:
        return False


def get(key):
    try:
        return config[key]
    except KeyError:
        return False

