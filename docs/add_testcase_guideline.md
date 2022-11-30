# Add Test Case Guideline
## Run Unit Test
You can use concole to launch test, run cmd in `scripts`
```bash
python tools/launch_pytest.py -v -s
```

## Add Test Case
You can add test case in `tests/test_cases`. `pytest` will run all files of the form test_*.py or *_test.py in the current directory and its subdirectories.

## HTML Visualize and Log
We support HTML format to visualize test case coverage and other information. And You should find more test information in `auto_test.log`.

## More Information
- [Pytest Document](https://docs.pytest.org/en/7.1.x/contents.html)