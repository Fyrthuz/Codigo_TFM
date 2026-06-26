import os
import tempfile
import pytest
from PIL import Image

from src.utils.dataset import LGGSegmentationDataset, SegmentationDataset


class TestLGGSegmentationDataset:
    @pytest.fixture
    def temp_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = os.path.join(tmpdir, "TCGA_001")
            os.makedirs(case_dir)
            img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            mask = Image.new("L", (256, 256), color=0)
            img.save(os.path.join(case_dir, "image.tif"))
            mask.save(os.path.join(case_dir, "image_mask.tif"))
            yield tmpdir

    def test_len(self, temp_data):
        ds = LGGSegmentationDataset(temp_data)
        assert len(ds) == 1

    def test_getitem_shape(self, temp_data):
        ds = LGGSegmentationDataset(temp_data)
        image, mask = ds[0]
        assert image.shape == (3, 256, 256)
        assert mask.shape == (1, 256, 256)

    def test_abstract_class_instantiation(self):
        with pytest.raises(TypeError):
            SegmentationDataset()

    def test_empty_directory(self, tmp_path):
        ds = LGGSegmentationDataset(str(tmp_path))
        assert len(ds) == 0

    def test_skip_missing_mask(self, temp_data):
        case_dir = os.path.join(temp_data, "TCGA_002")
        os.makedirs(case_dir)
        Image.new("RGB", (256, 256)).save(os.path.join(case_dir, "image2.tif"))
        ds = LGGSegmentationDataset(temp_data)
        assert len(ds) == 1
