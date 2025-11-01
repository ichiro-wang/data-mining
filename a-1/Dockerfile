FROM python:3.10-alpine

WORKDIR /myapp

COPY ./main.py ./node.py ./decision_tree.py /myapp/
COPY ./data /myapp/data

RUN pip install --no-cache-dir pandas numpy

ENTRYPOINT ["python", "main.py"]