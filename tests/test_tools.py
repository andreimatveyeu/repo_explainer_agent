import unittest
import os
import json
from repo_explainer_agent.tools import parse_python_file

class TestTools(unittest.TestCase):

    def test_parse_python_file(self):
        """
        Tests the parse_python_file tool with a sample Python file.
        """
        sample_file_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_python_file.py')
        parsed_data = parse_python_file(sample_file_path)

        # Define the expected parsed data structure
        expected_parsed_data = {
            "module_docstring": "This is a sample Python file for testing the parsing tool.",
            "imports": [
                {
                    "name": "import polars",
                    "lineno": 4,
                    "col_offset": 0,
                    "end_lineno": 4,
                    "end_col_offset": 13
                }
            ],
            "definitions": [
                {
                    "name": "sample_function",
                    "type": "function",
                    "docstring": "This is a sample function.\n\nArgs:\n    arg1: The first argument.\n    arg2: The second argument.\n\nReturns:\n    The sum of arg1 and arg2.",
                    "lineno": 6,
                    "col_offset": 0,
                    "end_lineno": 20,
                    "end_col_offset": 19
                },
                {
                    "name": "SampleClass",
                    "type": "class",
                    "docstring": "This is a sample class.",
                    "lineno": 22,
                    "col_offset": 0,
                    "end_lineno": 84,
                    "end_col_offset": 22,
                    "methods": [
                        {
                            "name": "__init__",
                            "type": "method",
                            "docstring": "The constructor for SampleClass.\n\nArgs:\n    name: The name of the instance.",
                            "lineno": 27,
                            "col_offset": 4,
                            "end_lineno": 36,
                            "end_col_offset": 24
                        },
                        {
                            "name": "sample_method",
                            "type": "method",
                            "docstring": "This is a sample method within SampleClass.\n\nArgs:\n    value: A value to process.\n\nReturns:\n    A string combining the instance name and the value.",
                            "lineno": 38,
                            "col_offset": 4,
                            "end_lineno": 51,
                            "end_col_offset": 61
                        },
                        {
                            "name": "sample_class_method",
                            "type": "method",
                            "docstring": "This is a sample class method.\n\nArgs:\n    data: Some class-level data.\n\nReturns:\n    A string indicating the class and data.",
                            "lineno": 54,
                            "col_offset": 4,
                            "end_lineno": 67,
                            "end_col_offset": 25
                        },
                        {
                            "name": "sample_static_method",
                            "type": "method",
                            "docstring": "This is a sample static method.\n\nArgs:\n    x: The first number.\n    y: The second number.\n\nReturns:\n    The product of x and y.",
                            "lineno": 70,
                            "col_offset": 4,
                            "end_lineno": 84,
                            "end_col_offset": 22
                        }
                    ]
                },
                {
                    "name": "another_function",
                    "type": "function",
                    "docstring": "Another simple function.",
                    "lineno": 86,
                    "col_offset": 0,
                    "end_lineno": 93,
                    "end_col_offset": 17
                }
            ],
            "error": None
        }

        # Assert that the parsed data matches the expected data
        # print(json.dumps(parsed_data, indent=2)) # Keep this commented out unless debugging
        self.assertEqual(parsed_data, expected_parsed_data)

if __name__ == '__main__':
    unittest.main()
