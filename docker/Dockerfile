FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import ViltForQuestionAnswering, ViltProcessor; \
c = 'dandelin/vilt-b32-finetuned-vqa'; \
ViltProcessor.from_pretrained(c); ViltForQuestionAnswering.from_pretrained(c)"

COPY main.py .
COPY src/ src/
COPY static/ static/
COPY templates/ templates/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
