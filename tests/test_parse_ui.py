import unittest

from parse_ui import export_elements


class ExportTests(unittest.TestCase):
    def test_preserves_ocr_caption_order_and_xyxy_coordinates(self):
        native = [
            {"content": "Add to cart", "bbox": [0.1, 0.2, 0.4, 0.3],
             "source": "box_ocr_content_ocr"},
            {"content": "A shopping cart icon", "bbox": [0.8, 0.1, 1.0, 0.2],
             "source": "box_yolo_content_yolo"},
        ]
        record = export_elements("web:6", native)
        self.assertEqual(record, {"sample_id": "web:6", "elements": [
            {"text": "Add to cart", "bbox": [0.1, 0.2, 0.4, 0.3]},
            {"text": "A shopping cart icon", "bbox": [0.8, 0.1, 1.0, 0.2]},
        ]})
        self.assertEqual(native[0]["source"], "box_ocr_content_ocr")
        self.assertEqual(export_elements("empty", []), {"sample_id": "empty", "elements": []})

    def test_rejects_invalid_coordinates_and_missing_content(self):
        for bbox in ([1, 2, 4, 3], [0, 0, float("nan"), 1],
                     [0, 0, float("inf"), 1], [-0.1, 0, 0.5, 1],
                     [0.4, 0.2, 0.1, 0.3], [0, 0, 0, 1], [0, 0, 1]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                export_elements("web:6", [{"content": "test", "bbox": bbox}])
        with self.assertRaisesRegex(ValueError, "no text/content"):
            export_elements("web:6", [{"content": None, "bbox": [0, 0, 1, 1]}])


if __name__ == "__main__":
    unittest.main()
