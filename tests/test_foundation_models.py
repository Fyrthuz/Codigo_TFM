import pytest
import torch

from src.models.foundation.base import FoundationModel, NoTrainingRequired
from src.models.foundation.universeg import UniVerSegModel


class TestFoundationBase:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            FoundationModel()

    def test_marker_mixin(self):
        assert issubclass(FoundationModel, NoTrainingRequired)


class TestUniVerSeg:
    def test_import_success(self):
        model = UniVerSegModel()
        assert isinstance(model, UniVerSegModel)

    def test_no_context_raises(self):
        model = UniVerSegModel()
        with pytest.raises(RuntimeError, match="No context set"):
            model(torch.randn(1, 3, 128, 128))

    def test_get_output_channels_default(self):
        model = UniVerSegModel()
        assert model.get_output_channels() == 1
