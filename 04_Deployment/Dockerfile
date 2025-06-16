FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

RUN mkdir -p output

CMD ["python", "starter.py", "2023", "03"]

#docker build -t ride-duration-prediction:v1 .
#docker run -it --rm ride-duration-prediction:v1
