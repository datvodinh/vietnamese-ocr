FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY  . /app

CMD [ "python", "src/hello.py"]
