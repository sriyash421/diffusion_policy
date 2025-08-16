import numpy as np
import time

def _clamp(value, limits):
    """Clamp value(s) between limits. Works with scalars and arrays."""
    lower, upper = limits
    if value is None:
        return None
    
    if isinstance(value, np.ndarray):
        result = value.copy()
        if upper is not None:
            result = np.minimum(result, upper)
        if lower is not None:
            result = np.maximum(result, lower)
        return result
    else:
        if (upper is not None) and (value > upper):
            return upper
        elif (lower is not None) and (value < lower):
            return lower
        return value

class PD(object):
    """A vectorized PD controller for multiple joints."""

    def __init__(
        self,
        num_joints,
        Kp=1.0,
        Kd=0.0,
        setpoint=None,
        sample_time=None,
        output_limits=None,
        auto_mode=True,
        proportional_on_measurement=False,
        differential_on_measurement=True,
        error_map=None,
        time_fn=None,
        starting_output=0.0,
    ):
        """
        Initialize a new vectorized PD controller.

        :param num_joints: Number of joints to control
        :param Kp: The value(s) for the proportional gain Kp (scalar or array)
        :param Kd: The value(s) for the derivative gain Kd (scalar or array)
        :param setpoint: The initial setpoint(s) that the PD will try to achieve (scalar or array)
        :param sample_time: The time in seconds which the controller should wait before generating
            a new output value. Set to None to disable sample time checking.
        :param output_limits: The initial output limits to use, given as an iterable with 2
            elements, for example: (lower, upper). Can be scalars or arrays.
        :param auto_mode: Whether the controller should be enabled (auto mode) or not (manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on
            the input directly rather than on the error
        :param differential_on_measurement: Whether the differential term should be calculated on
            the input directly rather than on the error
        :param error_map: Function to transform the error value in another constrained value
        :param time_fn: The function to use for getting the current time
        :param starting_output: The starting point for the PD's output (scalar or array)
        """
        self.num_joints = num_joints
        
        # Convert gains to arrays if they're scalars
        self.Kp = np.full(num_joints, Kp) if np.isscalar(Kp) else np.array(Kp)
        self.Kd = np.full(num_joints, Kd) if np.isscalar(Kd) else np.array(Kd)
        
        # Convert setpoint to array if it's scalar or None
        if setpoint is None:
            self.setpoint = np.zeros(num_joints)
        elif np.isscalar(setpoint):
            self.setpoint = np.full(num_joints, setpoint)
        else:
            self.setpoint = np.array(setpoint)
            
        self.sample_time = sample_time
        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        # Initialize arrays for PD terms
        self._proportional = np.zeros(num_joints)
        self._derivative = np.zeros(num_joints)

        self._last_time = None
        self._last_output = None
        self._last_error = None
        self._last_input = None

        if time_fn is not None:
            self.time_fn = time_fn
        else:
            try:
                self.time_fn = time.monotonic
            except AttributeError:
                self.time_fn = time.time

        self.output_limits = output_limits
        self.reset()

    def __call__(self, input_, dt=None):
        """
        Update the PD controller.

        :param input_: Current input values (array of length num_joints)
        :param dt: If set, uses this value for timestep instead of real time
        """
        if not self.auto_mode:
            return self._last_output if self._last_output is not None else np.zeros(self.num_joints)

        input_ = np.array(input_)
        
        now = self.time_fn()
        if dt is None:
            dt = now - self._last_time if (self._last_time is not None and now - self._last_time > 0) else 1e-16
        elif dt <= 0:
            raise ValueError('dt has negative value {}, must be positive'.format(dt))

        if self.sample_time is not None and dt < self.sample_time and self._last_output is not None:
            # Only update every sample_time seconds
            return self._last_output

        # Ensure arrays are initialized
        if self._proportional is None:
            self._proportional = np.zeros(self.num_joints)
        if self._derivative is None:
            self._derivative = np.zeros(self.num_joints)

        # Compute error terms
        error = self.setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)
        d_error = error - (self._last_error if (self._last_error is not None) else error)

        # Check if must map the error
        if self.error_map is not None:
            error = self.error_map(error)

        # Compute the proportional term
        if not self.proportional_on_measurement:
            # Regular proportional-on-error
            self._proportional = self.Kp * error
        else:
            # Proportional-on-measurement
            self._proportional = self._proportional - self.Kp * d_input

        # Compute derivative term
        if self.differential_on_measurement:
            self._derivative = -self.Kd * d_input / dt
        else:
            self._derivative = self.Kd * d_error / dt

        # Compute final output
        output = self._proportional + self._derivative

        # Apply output limits if specified
        if self.output_limits != (None, None):
            output = _clamp(output, self.output_limits)

        # Keep track of state
        self._last_output = output.copy()
        self._last_input = input_.copy()
        self._last_error = error.copy()
        self._last_time = now

        return output

    @property
    def components(self):
        """
        The P- and D-terms from the last computation as separate components as a tuple.
        """
        proportional = self._proportional.copy() if self._proportional is not None else np.zeros(self.num_joints)
        derivative = self._derivative.copy() if self._derivative is not None else np.zeros(self.num_joints)
        return proportional, derivative

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (Kp, Kd)."""
        return self.Kp.copy(), self.Kd.copy()

    @tunings.setter
    def tunings(self, tunings):
        """Set the PD tunings."""
        Kp, Kd = tunings
        self.Kp = np.full(self.num_joints, Kp) if np.isscalar(Kp) else np.array(Kp)
        self.Kd = np.full(self.num_joints, Kd) if np.isscalar(Kd) else np.array(Kd)

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not."""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PD controller."""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        Enable or disable the PD controller, optionally setting the last output value.
        
        :param enabled: Whether auto mode should be enabled, True or False
        :param last_output: The last output values (array), that the PD should start from
        """
        if enabled and not self._auto_mode:
            # Switching from manual mode to auto, reset
            self.reset()

        self._auto_mode = enabled

    @property
    def output_limits(self):
        """
        The current output limits as a 2-tuple: (lower, upper).
        """
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Set the output limits."""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        # Handle scalar limits
        if min_output is not None and np.isscalar(min_output):
            min_output = np.full(self.num_joints, min_output)
        if max_output is not None and np.isscalar(max_output):
            max_output = np.full(self.num_joints, max_output)

        if (min_output is not None and max_output is not None and 
            np.any(max_output < min_output)):
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        if self._last_output is not None:
            self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        """
        Reset the PD controller internals.
        """
        self._proportional = np.zeros(self.num_joints)
        self._derivative = np.zeros(self.num_joints)

        self._last_time = self.time_fn()
        self._last_output = None
        self._last_input = None
        self._last_error = None 