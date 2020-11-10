FROM python:3.8.1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /code

COPY ./ ./

# ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" 