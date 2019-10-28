def isregular(material):
    if 'permfunc' in dir(material):
        return 0

    return 1