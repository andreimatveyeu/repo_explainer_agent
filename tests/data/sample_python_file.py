"""
This is a sample Python file for testing the parsing tool.
"""
import polars

def sample_function(arg1, arg2):
    """
    This is a sample function.

    Args:
        arg1: The first argument.
        arg2: The second argument.

    Returns:
        The sum of arg1 and arg2.
    """
    # Add some random code
    temp_sum = arg1 + arg2
    print(f"Calculating sum: {temp_sum}")
    return temp_sum

class SampleClass:
    """
    This is a sample class.
    """

    def __init__(self, name):
        """
        The constructor for SampleClass.

        Args:
            name: The name of the instance.
        """
        # Add some random code
        print(f"Initializing SampleClass with name: {name}")
        self.name = name

    def sample_method(self, value):
        """
        This is a sample method within SampleClass.

        Args:
            value: A value to process.

        Returns:
            A string combining the instance name and the value.
        """
        # Add some random code
        processed_value = value * 2
        print(f"Processing value: {processed_value}")
        return f"Name: {self.name}, Value: {processed_value}"

    @classmethod
    def sample_class_method(cls, data):
        """
        This is a sample class method.

        Args:
            data: Some class-level data.

        Returns:
            A string indicating the class and data.
        """
        # Add some random code
        class_info = f"Class: {cls.__name__}, Data: {data}"
        print(f"Using class method with info: {class_info}")
        return class_info

    @staticmethod
    def sample_static_method(x, y):
        """
        This is a sample static method.

        Args:
            x: The first number.
            y: The second number.

        Returns:
            The product of x and y.
        """
        # Add some random code
        product = x * y
        print(f"Calculating product: {product}")
        return product

def another_function():
    """
    Another simple function.
    """
    # Add some random code
    print("Executing another function")
    result = "done"
    return result
