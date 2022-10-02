import csv

from core import getCurrentModel, getFutureModel


class Offline:
    models = []

    def steps(self):
        getCurrentModel()
        self.read_models()

    def insert_model(self, model):
        with open('pool/models.csv', 'w') as file:
            writer = csv.writer(file)
            if len(self.models) == 0:
                self.models.append(['X1', 'X2', 'SD', 'MEAN', 'MIN', 'MAX'])
            self.models.append(model)
            writer.writerows(self.models)

    def read_models(self):
        self.models = []
        with open('pool/models.csv', 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                self.models.append(row)

    def get_similar(self, historical,data):
        new_wind = self.get_statistics(data)
        self.read_models()
        similarity = self.euclidean_distance(new_wind, self.models[0])
        similar = self.models[0]
        for l in self.models:
            s = self.euclidean_distance(new_wind, l)
            if similarity > s:
                similarity = s
                similar = l
        return similar

    def euclidean_distance(x, y):
        from math import pow, sqrt
        return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    euclidean_distance([0, 3, 4, 5], [7, 6, 3, -1])

    def get_statistics(self, data):
        total_sd = 0
        total_min = 0
        total_max = 0
        total_mean = 0
        total_h_mean = 0
        length = len(data)
        import statistics
        for d in data:
            total_sd += statistics.stdev(d)
            total_mean += statistics.mean(d)
            total_h_mean += statistics.harmonic_mean(d)
            total_max += max(d)
            total_min += min(d)

        return [total_sd / float(length), total_mean / float(length), total_h_mean / float(length),
                total_min / float(length), total_max / float(length)]


if __name__ == '__main__':
    off = Offline()
    off.read_models()
    off.insert_model([2, 4, 3, 4, 2, 3])
    print(off.models)
    off.get_statistics([[1, 3, 4], [3, 4, 5]])
    off.insert_model(off.get_statistics([[1, 3, 4], [3, 4, 5]]))
