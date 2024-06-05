FROM python:3.10.12

COPY . /app
WORKDIR /app


RUN pip install -r requirements.txt


RUN python -c "import nltk; \
nltk.download('punkt'); \
nltk.download('averaged_perceptron_tagger'); \
nltk.download('maxent_ne_chunker'); \
nltk.download('words')"

CMD [ "python3","pan_submission.py" ]