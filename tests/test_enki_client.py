import pytest
from enki_client.client import EnkiClient

class TestEnkiClient:
    @pytest.fixture
    def client(self, mocker):
        api_key = 'test_api_key'
        client = EnkiClient(api_key)
        return client

    def test_use_model(self, client, mocker):
        model = mocker.Mock()
        model.endpoint = 'test_endpoint'
        client.use_model(model)
        assert client.model == model
        assert client.kserve_client is not None

    def test_infer(self, client, mocker):
        model = mocker.Mock()
        model.endpoint = 'test_endpoint'
        client.use_model(model)

        request = mocker.Mock()
        inference_request = mocker.Mock()
        mocker.patch.object(model, 'to_inference_request', return_value=inference_request)

        inference_response = mocker.Mock()
        mocker.patch.object(client.kserve_client, 'infer', return_value=inference_response)

        response = mocker.Mock()
        mocker.patch.object(model, 'to_response', return_value=response)

        assert client.infer(request) == response