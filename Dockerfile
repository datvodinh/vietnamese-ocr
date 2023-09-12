FROM python:3.10
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
# ADD ./src/main.py .
# CMD [ "python","./main.py" ]
