from main import app
import serverless_wsgi

def handler(event, context):
    import json
    print("EVENT RECEIVED:", json.dumps(event))
    return serverless_wsgi.handle_request(app, event, context)
