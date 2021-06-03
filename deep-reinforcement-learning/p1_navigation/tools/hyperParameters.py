from math import log, exp

class Parameters:

    def __init__(self, value, lower_bound, upper_bound):
        self._value = value
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._steps = 0

    def _updateValue(self, value):
        #print("NEW", self._value)
        self._value = value

    def value(self):
        #print(self._value)
        return self._value

    def next(self):
        raise NotImplementedError('Parameters::next to be implemented')


class geometricParameter(Parameters):

    def __init__(self, initialValue, factor=0.99, steps=None, direction='down', lower_bound=0.01, upper_bound=1.):
        super(geometricParameter, self).__init__(initialValue, lower_bound, upper_bound)
        self.factor = factor
        self.steps = steps
        if steps is not None:
            target = lower_bound if direction=='down' else upper_bound
            self.factor = exp((log(target) - log(initialValue))/float(steps))

    def next(self):
        if (self._value >= self._lower_bound) and (self._value <= self._upper_bound):
            new_value = min(max(self._value*self.factor, self._lower_bound),self._upper_bound)
            self._updateValue(new_value)
        self._steps = self._steps + 1
        return self.value()


class linearParameter(Parameters):

    def __init__(self, initialValue, increment=-0.01, steps=None, direction='down', lower_bound=0., upper_bound=1.):
        super(linearParameter, self).__init__(initialValue, lower_bound, upper_bound)
        self.increment = increment
        self.steps = steps
        if steps is not None:
            target = lower_bound if direction=='down' else upper_bound
            self.increment = (target-initialValue)/float(steps)

    def next(self):
        if (self._value >= self._lower_bound) and (self._value <= self._upper_bound):
            new_value = min(max(self._value+self.increment, self._lower_bound),self._upper_bound)
            self._updateValue(new_value)
        self._steps = self._steps + 1
        return self.value()

class Hyperparameters():
    '''
    This class allows you to create hyperparmaters that can vary in turns following a predefined profil
    Input :
        - profil =
         'geometric' -> Value(n+1) = Value(n) * decay
         'linear'    -> Value(n+1) = Value(n) + decay
         'constant'  
        - init = Starting value 
        - bound = Ending value 
    
        Either 
        - decay = Decay rate if steps is not used (Default 0.99)
        - steps = Number of turns to move from init to bound (Default None), therfore decay is calculated as follows 
           for 'linear'   : decay = (bound-init)/steps
           for 'geometric': decay = exp((log(bound) - log(init))/steps)
    '''

    def __init__(self, name="", profil='geometric', init=0., bound=1., steps=None, decay=0.99):

        self.value = init
        self.name = name
        if profil == 'constant':
            self._profil = self._constant
        if profil == 'linear':
            self._profil = self._linear
        if profil == 'geometric':
            self._profil = self._geometric
        self._direction = 'up' if init < bound else 'down'
        self._init = init
        self._bound = bound
        self._delimitor = 'steps' if steps is not None else 'decay'
        self._delimitor_step = steps if steps is not None else decay

        if steps is not None:
            if profil == 'constant':
                self.decay = 0
            if profil == 'linear':
                self.decay = (bound-init)/float(steps)
            if profil == 'geometric':
                self.decay = exp((log(bound) - log(init))/float(steps))

    def _constant(self):
        pass

    def _linear(self):
        self.value = self.value + self.decay

    def _geometric(self):
        self.value = self.value * self.decay    

    def __repr__(self):
        return "Parameter {s.name}={s.value} profil {s._profil} going {s._direction} with {s._delimitor}={s._delimitor_step}".format(s=self)

    def display(self):
        return "{s.name}={s.value:.1e}".format(s=self)

    def next(self):
        '''
        Updates the parameters value according to profil
        '''
        if ((self._direction == 'up') and (self.value < self._bound)) or ((self._direction == 'down') and (self.value > self._bound)):
            self._profil()
        else:
            self.value = self._bound
