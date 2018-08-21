import sys 

class BaseScheduler(object):
    def __init__(self, init_epsilon, min_epsilon, schedule_timesteps, **kwargs):
        self.init_epsilon = float(init_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.schedule_timesteps = int(schedule_timesteps)
    
    def value(self, t):
        raise NotImplementedError
    
    def end_value(self):
        raise NotImplementedError

class LinearScheduler(BaseScheduler):
    def value(self, t):
        """ Value of the schedule at time t"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.init_epsilon + fraction * (self.min_epsilon - self.init_epsilon)

    def end_value(self):
        return self.value(self.schedule_timesteps)

def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        raise NotImplementedError("Unknown scheduler: {}".format(string))

if __name__ == "__main__":
    sch = LinearScheduler(1.0, 0.0, 10)

    for t in range(10):
        print('value', sch.value(t))
