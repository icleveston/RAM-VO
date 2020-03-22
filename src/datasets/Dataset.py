import abc


class Dataset:

    @abc.abstractmethod
    def next_frame(self):
        pass

    @abc.abstractmethod
    def ground_truth(self):
        pass
