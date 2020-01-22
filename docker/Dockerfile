FROM python:3.7-slim

COPY . ../ratel
WORKDIR /ratel

RUN pip install pipenv
RUN pipenv lock --requirements > requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "/ratel/bernoulli_bandit.py"]