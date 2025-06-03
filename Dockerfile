FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y postgresql-devel gcc python3-devel
RUN yum install postgresql-devel
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --upgrade psycopg2-binary
COPY ./*.py ./

CMD ["lambda_handler.handler"]