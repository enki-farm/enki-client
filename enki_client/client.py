import kserve

class EnkiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        self.kserve_client = None
        
    def use_model(self, model):
        self.model = model
        self.kserve_client = kserve.InferenceServerClient(self.model.endpoint)
    
    def infer(self, request):
        inference_request = self.model.to_inference_request(request)
        inference_response = self.kserve_client.infer(inference_request)

        return self.model.to_response(inference_response)