FROM python:3.7-slim

COPY . ../ratel

RUN pip install numpy pandas
RUN cd ./ratel

ENTRYPOINT ["./bernoulli_bandit.py"]