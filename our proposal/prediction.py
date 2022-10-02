from core import updateCurrentModel


class Prediction:
    def kswin_algorithm(self,data_stream):
        from river import drift
        from offline_learning import Offline
        drifts = []
        kswin = drift.KSWIN(alpha=0.0001, seed=42)
        print('------------(Kolmogorov-Smirnov Windowing method)------------')
        for i, x in enumerate(data_stream):
            _ = kswin.update(x)
            if kswin.change_detected:
                print(f"Change detected at index {i}")
                off = Offline()
                similar = off.get_similar(x)
                if similar > 0.5:
                    updateCurrentModel(similar)
                else:
                    ''' retain model '''

                drifts.append(i)
        # plot_data(dist_a, dist_b, dist_c, drifts=drifts, name="Kolmogorov-Smirnov Windowing method")

    def invokeOnlineAdaptation(self):
        future_model = self.getFutureModel()
        current_model = self.getCurrentModel()
        bestModel = self.compareBetweenCurrentAndFutureModel(current_model, future_model)
        if bestModel == future_model:
            self.updateCurrentByFuture(bestModel)

    def compareBetweenCurrentAndFutureModel(self, current, future):
        if future > current:
            print('future Model')
            return future
        else:
            print('current')
            return current

    def updateCurrentByFuture(self, future):
        print('update Model')


if __name__ == '__main__':
    s = Step1()
    s.invokeOnlineAdaptation()
