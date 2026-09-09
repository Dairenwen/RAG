FROM python:3.12-slim

WORKDIR /app

COPY Project1/deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY Project1 /app/Project1
COPY Src /app/Src

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "Project1.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
