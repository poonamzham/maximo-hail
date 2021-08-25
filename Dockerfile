FROM python:3.7-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

ENV HOME="/usr/src/app"

COPY requirements.txt .

RUN pip install opencv-python==4.5.3.56
RUN pip install -r requirements.txt

EXPOSE 8501

COPY app.py helper.py ./
RUN mkdir -p .streamlit

ENTRYPOINT ["streamlit", "run", "--server.headless=true"]

CMD ["app.py"]
