from __future__ import annotations

import unittest

from transfer_learning.visdrone import (
    CLASS_NAMES,
    COCO_CLASS_IDS,
    parse_annotation,
    sample_by_sequence,
    sequence_id,
    to_yolo_lines,
)


class ParseAnnotationTests(unittest.TestCase):
    def test_maps_visdrone_categories_to_target_classes(self):
        text = "\n".join([
            "10,20,30,40,1,1,0,0",   # pedestrian -> person
            "10,20,30,40,1,2,0,0",   # people -> person
            "10,20,30,40,1,3,0,0",   # bicycle
            "10,20,30,40,1,4,0,0",   # car
            "10,20,30,40,1,5,0,0",   # van -> car
            "10,20,30,40,1,6,0,0",   # truck
            "10,20,30,40,1,9,0,0",   # bus
            "10,20,30,40,1,10,0,0",  # motor -> motorcycle
        ])
        boxes, masks = parse_annotation(text)
        names = [CLASS_NAMES[box[0]] for box in boxes]
        self.assertEqual(names, ["person", "person", "bicycle", "car", "car", "truck", "bus", "motorcycle"])
        self.assertEqual(masks, [])

    def test_ignored_regions_and_unmapped_categories_become_masks(self):
        text = "\n".join([
            "1,2,3,4,0,0,0,0",    # ignored region (score 0)
            "5,6,7,8,1,7,0,0",    # tricycle
            "9,9,9,9,1,8,0,0",    # awning-tricycle
            "1,1,5,5,1,11,0,0",   # others
            "2,2,6,6,0,4,0,0",    # car flagged as ignored by score 0
        ])
        boxes, masks = parse_annotation(text)
        self.assertEqual(boxes, [])
        self.assertEqual(masks, [(1, 2, 3, 4), (5, 6, 7, 8), (9, 9, 9, 9), (1, 1, 5, 5), (2, 2, 6, 6)])

    def test_skips_degenerate_boxes_and_blank_lines(self):
        boxes, masks = parse_annotation("10,10,0,5,1,4,0,0\n\n10,10,5,5,1,4,0,0,\n")
        self.assertEqual(boxes, [(2, 10, 10, 5, 5)])
        self.assertEqual(masks, [])


class YoloLinesTests(unittest.TestCase):
    def test_normalizes_center_format(self):
        lines = to_yolo_lines([(2, 100, 50, 200, 100)], img_w=1000, img_h=500)
        self.assertEqual(lines, ["2 0.200000 0.200000 0.200000 0.200000"])

    def test_clips_boxes_to_image_bounds(self):
        lines = to_yolo_lines([(0, -10, 450, 30, 100)], img_w=1000, img_h=500)
        self.assertEqual(lines, ["0 0.010000 0.950000 0.020000 0.100000"])

    def test_class_override_for_coco_ids(self):
        lines = to_yolo_lines([(5, 0, 0, 10, 10)], img_w=10, img_h=10, class_map=COCO_CLASS_IDS)
        self.assertTrue(lines[0].startswith("7 "))


class SamplingTests(unittest.TestCase):
    def test_sequence_id_is_first_token(self):
        self.assertEqual(sequence_id("0000002_00005_d_0000014"), "0000002")

    def test_sampling_spreads_across_sequences_and_is_deterministic(self):
        stems = [f"{seq:07d}_{i:05d}_d_0000001" for seq in range(4) for i in range(10)]
        picked = sample_by_sequence(stems, 8, seed=0)
        self.assertEqual(len(picked), 8)
        self.assertEqual({sequence_id(s) for s in picked}, {"0000000", "0000001", "0000002", "0000003"})
        self.assertEqual(picked, sample_by_sequence(stems, 8, seed=0))

    def test_sampling_returns_everything_when_n_exceeds_population(self):
        stems = ["0000001_00001_d_1", "0000002_00001_d_1"]
        self.assertEqual(sorted(sample_by_sequence(stems, 10, seed=0)), sorted(stems))


if __name__ == "__main__":
    unittest.main()
