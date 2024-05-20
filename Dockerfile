# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.10
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user/ \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/movieapp
COPY --chown=user . $HOME/movieapp/
COPY ./requirements.txt ~/movieapp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

CMD ["chainlit", "run", "app.py", "--port", "7860"]
