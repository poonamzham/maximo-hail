FROM python:3.7-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN mkdir -p .streamlit

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install opencv-python==4.5.3.56
RUN pip install -r requirements.txt

EXPOSE 8501

RUN mkdir -p /streamlit
COPY app.py helper.py ./

ENTRYPOINT ["streamlit", "run", "--server.headless=true"]

CMD ["app.py"]
