FROM python:3.7

WORKDIR /usr/src/app

COPY . .

RUN python3 -m pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "application.py"]
