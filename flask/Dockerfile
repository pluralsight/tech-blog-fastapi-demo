FROM python:3.7-slim

WORKDIR /opt/app

COPY ./requirements.txt ./requirements.txt
COPY ./script_requirements.txt ./script_requirements.txt

RUN pip install -r requirements.txt
RUN pip install -r script_requirements.txt
