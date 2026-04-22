FROM python:3.11-slim
WORKDIR /workspace
COPY . /workspace
CMD ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
