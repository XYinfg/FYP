from abc import ABC, abstractmethod

class SchedulerBase(ABC):
    @abstractmethod
    def schedule(self, tasks, machines, current_time):
        pass
