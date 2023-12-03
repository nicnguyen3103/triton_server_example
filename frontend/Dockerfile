FROM python:3.10.12

WORKDIR /code
# for open cv support

COPY . /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir ~/.streamlit

COPY config.toml ~/.streamlit/config.toml

CMD ["streamlit", "run", "streamlit.py"]
