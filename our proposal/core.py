def getFutureModel():
    f = open("pool/future", "r")
    current = f.read()
    return int(current.__str__())


def getCurrentModel():
    f = open("pool/current", "r")
    current = f.read()
    return int(current.__str__())


def updateCurrentModel(data):
    f = open("pool/current", "w")
    current = f.write(data)
    return current
