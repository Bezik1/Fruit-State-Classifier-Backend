FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

EXPOSE 8000

CMD ["conda", "run", "-n", "pytorch", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]