# install python --> base layer 
FROM python:3.10-slim 

# set our workdir 
WORKDIR /app/

# copy the files 
COPY ./frontend/ .
COPY ./fast_api/ .

# install packages 
RUN pip install --no-cache-dir -r requirements.txt

# copy the dataset required 
COPY ./data/raw/ ./data/

# final cmd statement --> run the container 
CMD ["streamlit" ,"run" , "app.py"]

