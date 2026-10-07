# install python --> base layer 
FROM python:3.12-slim 

# set our workdir 
WORKDIR /app/

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# copy the files 
COPY . .

# install packages 
RUN pip install --no-cache-dir -r requirements.txt

# expose
EXPOSE 8000

# final cmd statement --> run the container 

CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]