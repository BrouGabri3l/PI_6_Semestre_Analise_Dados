from main import app

def handler(event, context):
    from awslambdaric.lambda_context import LambdaContext
    from mangum import Mangum

    asgi_handler = Mangum(app)
    return asgi_handler(event, context)