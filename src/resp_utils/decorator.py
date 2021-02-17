def synchronize(func):
    def wrapper(self, *args, **kwargs):
        self.mutex.lock()
        ret = func(self, *args, **kwargs)
        self.mutex.unlock()
        return ret

    return wrapper
