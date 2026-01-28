import math


class Template(object):
    def sum(self, value1, value2) -> float:
        """
        Sum two float values.

        Parameters:
        - value1: The first value to be summed. It can be an int, float, or string.
        - value2: The second value to be summed. It can be an int, float, or string.

        Returns:
        The sum of the two values.
        """
        if isinstance(value1, str):
            value1 = float(value1)
        if isinstance(value2, str):
            value2 = float(value2)
        return math.fsum([value1, value2])
