FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y postgresql-devel gcc python3-devel

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./*.py ./

CMD ["lambda_handler.handler"]