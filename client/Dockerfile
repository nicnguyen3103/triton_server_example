FROM python:3.10.12

WORKDIR /code
# for open cv support
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app


CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "5000"]
