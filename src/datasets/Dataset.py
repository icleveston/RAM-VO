import abc


class Dataset:

    @abc.abstractmethod
    def next_frame(self):
        pass

    @abc.abstractmethod
    def evaluate(self, trajectory):
        pass
