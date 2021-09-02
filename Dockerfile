FROM python:3.7-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /work/
RUN chown 1001 /work \
    && chmod "g+rwX" /work \
    && chown 1001:root /work

ENV HOME='/work'

COPY requirements.txt .

RUN pip install opencv-python==4.5.3.56
RUN pip install -r requirements.txt

EXPOSE 8501

COPY --chown=1001:root app.py helper.py ./

USER 1001

ENTRYPOINT ["streamlit", "run", "--server.headless=true"]

CMD ["app.py"]
