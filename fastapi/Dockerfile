FROM python:3.7-slim

WORKDIR /opt/app

RUN \
    apt-get update \
    && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
