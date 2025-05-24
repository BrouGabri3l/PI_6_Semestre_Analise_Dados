FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./*.py ./

CMD ["lambda_handler.handler"]