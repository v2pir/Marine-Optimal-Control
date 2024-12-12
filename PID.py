class Controller:

    def __init__(self, kp, ki, kd):
        self.kp = kp #p gain
        self.ki= ki #i gain
        self.kd = kd #d gain
        self.totalError = 0
        self.prevError = 0

    def output(self, currentTilt, setpoint):

        error = setpoint - currentTilt

        proportionalError = self.kp * error

        self.totalError += error
        integralError = self.ki * self.totalError

        derivativeError = self.kd * (error - self.prevError)
        self.prevError = error

        return proportionalError + integralError + derivativeError
